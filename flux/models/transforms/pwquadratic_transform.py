import numpy as np
import torch
import typing as t

from torch import Tensor as tt

from flux.models.transforms.base import BaseCouplingTransform


def modified_softmax (v,w):
    v=torch.exp(v)
    vsum=torch.cumsum(v, axis=-1)
    vnorms=torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)
    vnorms_tot=vnorms[:, :, -1].clone() 
    return torch.div(v,torch.unsqueeze(vnorms_tot,axis=-1)) 


class PWQuadraticCouplingTransform(BaseCouplingTransform):
    # TODO: implement this
    @staticmethod
    def forward(x: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        logj = None

        v_tilde=theta[:,:,:int(np.ceil(theta.shape[2]/2))]
        w_tilde=theta[:,:,v_tilde.shape[2]:]
        N, k, b = w_tilde.shape
        Nx, kx = x.shape
        assert N == Nx and k == kx, "Shape mismatch"

        w=torch.exp(w_tilde)
        wsum = torch.cumsum(w, axis=-1) 
        wnorms = torch.unsqueeze(wsum[:, :, -1], axis=-1) 
        w = w/wnorms
        wsum=wsum/wnorms
        wsum_shift=torch.cat((torch.zeros([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype),wsum),axis=-1)

        v=modified_softmax(v_tilde, w)

        #tensor of shape (N,k,b+1) with 0 entry if x is smaller than the cumulated w and 1 if it is bigger
        #this is used to find the bin with the number mx in which x lies; for this, the sum of the bin 
        #widths w has to be smaller to the left and bigger to the right
        finder=torch.where(wsum>torch.unsqueeze(x,axis=-1),torch.zeros_like(wsum),torch.ones_like(wsum))
        eps = torch.finfo(wsum.dtype).eps
        #the bin number can be extracted by finding the last index for which finder is nonzero. As wsum 
        #is increasing, this can be found by searching for the maximum entry of finder*wsum. In order to 
        #get the right result when x is in the first bin and finder is everywhere zero, a small first entry 
        #is added
        mx=torch.unsqueeze(  #we need to unsqueeze for later operations
            torch.argmax( #we search for the maximum in order to find the last bin for which x was greater than wsum
                torch.cat((torch.ones([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype)*eps,finder*wsum),
                            axis=-1),  #we add an offset to ensure that if x is in the first bin, a maximal argument is found 
                axis=-1), 
            axis=-1)

        # x is in the mx-th bin: x \in [0,1],
        # mx \in [[0,b-1]], so we clamp away the case x == 1
        mx = torch.clamp(mx, 0, b - 1).to(torch.long)
        # Need special error handling because trying to index with mx
        # if it contains nans will lock the GPU. (device-side assert triggered)
        # if torch.any(torch.isnan(mx)).item() or torch.any(mx < 0) or torch.any(mx >= b):
        #     raise AvertedCUDARuntimeError("NaN detected in PWQuad bin indexing")

        # alpha (element of [0.,1], the position of x in its bin)
        # gather collects the cumulated with of all bins until the one in which x lies
        # alpha=(x- Sum_(k=0)^(i-1) w_k)/w_b for x in bin b
        alphas=torch.div((x-torch.squeeze(torch.gather(wsum_shift,-1,mx),axis=-1)),
                                torch.squeeze(torch.gather(w,-1,mx),axis=-1))

        #vw_i= (v_i+1 - v_i)w_i/2 where i is the bin index
        vw=torch.cat((torch.zeros([v.shape[0],v.shape[1],1]).to(wsum.device, wsum.dtype),
                                        torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)),axis=-1)

        #quadratic term
        out_1=torch.mul((alphas**2)/2,torch.squeeze(torch.mul(torch.gather(v,-1, mx+1)-torch.gather(v,-1, mx),
                                                            torch.gather(w,-1,mx)),axis=-1))

        #linear term
        out_2=torch.mul(torch.mul(alphas,torch.squeeze(torch.gather(v,-1,mx),axis=-1)),
                                torch.squeeze(torch.gather(w,-1,mx),axis=-1))

        #constant
        out_3= torch.squeeze(torch.gather(vw,-1,mx),axis=-1)


        out=out_1+out_2+out_3

        #the derivative of this transformation is the linear interpolation between v_i-1 and v_i at alpha
        #the jacobian is the product of all linear interpolations
        if compute_log_jacobian:
            logj=torch.squeeze(
                torch.log(torch.unsqueeze(
                        torch.prod(#we need to take the product over all transformed dimensions
                            torch.lerp(torch.squeeze(torch.gather(v,-1,mx),axis=-1),
                                        torch.squeeze(torch.gather(v,-1,mx+1),axis=-1),alphas),
                            #linear extrapolation between alpha, mx and mx+1
                        axis=-1),
                    axis=-1)),
                axis=-1)
            
        # Regularization: points must be strictly within the unit hypercube
        # Use the dtype information from pytorch
        eps = torch.finfo(out.dtype).eps
        out = out.clamp(
            min=eps,
            max=1. - eps
        )
        return out, logj

    @staticmethod
    def backward(y: tt, theta: tt, compute_log_jacobian: bool = True) -> t.Tuple[tt, t.Optional[tt]]:
        logj = None

        # TODO do a bottom-up assesment of how we handle the differentiability of variables
        
        v_tilde=theta[:,:,:int(np.ceil(theta.shape[2]/2))]
        w_tilde=theta[:,:,v_tilde.shape[2]:]
        N, k, b = w_tilde.shape
        
        Nx, kx = y.shape
        assert N == Nx and k == kx, "Shape mismatch"
        
        w=torch.exp(w_tilde)
        wsum = torch.cumsum(w, axis=-1) 
        wnorms = torch.unsqueeze(wsum[:, :, -1], axis=-1) 
        w = w/wnorms
        wsum=wsum/wnorms
        wsum_shift=torch.cat((torch.zeros([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype),wsum),axis=-1)
        
        v=modified_softmax(v_tilde, w)
        
        #need to find the bin number for each of the y/x
        #-> find the last bin such that y is greater than the constant of the quadratic equation
        
        #vw_i= (v_i+1 - v_i)w_i/2 where i is the bin index. VW is the constant of the quadratic equation
        vw=torch.cat((torch.zeros([v.shape[0],v.shape[1],1]).to(wsum.device, wsum.dtype),
                                    torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)),axis=-1)
        # finder is contains 1 where y is smaller then the constant and 0 if it is greater
        finder=torch.where(vw>torch.unsqueeze(y,axis=-1),torch.zeros_like(vw),torch.ones_like(vw))
        eps = torch.finfo(vw.dtype).eps
        #the bin number can be extracted by finding the last index for which finder is nonzero. As vw 
        #is increasing, this can be found by searching for the maximum entry of finder*vw. In order to 
        #get the right result when y is in the first bin and finder is everywhere zero, a small first entry 
        #is added and mx is reduced by one to account for the shift.
        mx=torch.unsqueeze(
            torch.argmax(#we search for the maximum in order to find the last bin for which y was greater than vw
                torch.cat((torch.ones([vw.shape[0],vw.shape[1],1]).to(vw.device, vw.dtype)*eps,finder*(vw+1)),axis=-1),
                axis=-1), #we add an offset to ensure that if x is in the first bin, a maximal argument is found
            axis=-1)-1 # we substract -1 to account for the offset
        
        # x is in the mx-th bin: x \in [0,1],
        # mx \in [[0,b-1]], so we clamp away the case x == 1
        edges = torch.clamp(mx, 0, b - 1).to(torch.long)
        
        # Need special error handling because trying to index with mx
        # if it contains nans will lock the GPU. (device-side assert triggered)
        # if torch.any(torch.isnan(edges)).item() or torch.any(edges < 0) or torch.any(edges >= b):
        #     raise AvertedCUDARuntimeError("NaN detected in PWQuad bin indexing")
        
        #solve quadratic equation
        
        #prefactor of quadratic term
        a=torch.squeeze(torch.mul(torch.gather(v,-1, edges+1)-torch.gather(v,-1, edges),
                                                            torch.gather(w,-1,edges)),axis=-1)
        #prefactor of linear term
        b=torch.mul(torch.squeeze(torch.gather(v,-1,edges),axis=-1),torch.squeeze(torch.gather(w,-1,edges),axis=-1))
        #constant - y
        c= torch.squeeze(torch.gather(vw,-1,edges),axis=-1)-y
        
        #ensure that division by zero is taken care of
        eps = torch.finfo(a.dtype).eps
        a=torch.where(torch.abs(a)<eps,eps*torch.ones_like(a),a)
        
        d = (b**2) - (2*a*c)
        
        assert not torch.any(d<0), "Value error in PWQuad inversion"
        assert not torch.any(a==0), "Value error in PWQuad inversion, a==0"
        
        # find two solutions
        sol1 = (-b-torch.sqrt(d))/(a)
        sol2 = (-b+torch.sqrt(d))/(a)
        
        # choose solution which is in the allowed range
    
        sol=torch.where((sol1>=0)&(sol1<1), sol1, sol2)
        
        # if torch.any(torch.isnan(sol)).item():
        #     raise AvertedCUDARuntimeError("NaN detected in PWQuad inversion")
        
        eps = torch.finfo(sol.dtype).eps
        
        
        sol = sol.clamp(
            min=eps,
            max=1. - eps
        )
        
        #the solution is the relative position inside the bin. This can be
        #converted to the absolute position by adding the sum of the bin widths up to this bin
        x=torch.mul(torch.squeeze(torch.gather(w,-1,edges),axis=-1), sol)+torch.squeeze(torch.gather(wsum_shift,-1,edges),axis=-1)
        
        eps = torch.finfo(x.dtype).eps
        
        x = x.clamp(
            min=eps,
            max=1. - eps
        )
        
        if compute_log_jacobian:
            logj =-torch.squeeze(torch.log(
                torch.unsqueeze(torch.prod(#we have to take the product of the jacobian of all dimensions
                    torch.lerp(torch.squeeze(torch.gather(v,-1,edges),axis=-1),torch.squeeze(torch.gather(v,-1,edges+1),
                                                                                            axis=-1),sol),
                    axis=-1), #linear extrapolation between sol, edges and edges+1 gives the jacobian of the forward transformation. The prefactor of -1 is the log of the jacobian of the inverse
                axis=-1)),
            axis=-1)
        
        return x.detach(), logj

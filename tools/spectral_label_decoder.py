"""Small shrunk pair-log-lift decoder; learned only from training targets."""
import numpy as np

def pair_lift(y,prior_strength=50.):
 y=np.asarray(y,float);n=len(y);p=(y.sum(axis=0)+1)/(n+2);ind=p[:,None]*p[None,:];posterior=(y.T@y+prior_strength*ind)/(n+prior_strength);J=np.clip(np.log(posterior/ind),-2,2);np.fill_diagonal(J,0);return J

def decode(scores,interaction,strength):
 s=np.asarray(scores,float)
 if strength==0:return (s>=.5).astype(int)
 k=s.shape[1];sets=((np.arange(2**k)[:,None]>>np.arange(k))&1).astype(float);logit=np.log(np.clip(s,1e-8,1-1e-8)/(1-np.clip(s,1e-8,1-1e-8)));prior=.5*strength*np.einsum('bi,ij,bj->b',sets,interaction,sets);utility=logit@sets.T+prior[None,:];return sets[np.argmax(utility,axis=1)].astype(int)

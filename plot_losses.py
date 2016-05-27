import matplotlib.pyplot as plt
import numpy as np

x=(np.array(range(-100,200,1))/100.).tolist()
hinge=[max(0,-xx-0+1) for xx in x]
plt.plot(x,np.array(hinge))
plt.plot(x,np.square(np.array(hinge)))

x=np.array(x)
g=np.exp(x)
g=g/np.sum(g)

g=-np.log(1./(1+np.exp(-x)))


plt.plot(x,g)
plt.legend
plt.savefig('foo.png', bbox_inches='tight')
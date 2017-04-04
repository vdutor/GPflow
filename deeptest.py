from matplotlib import pyplot as plt
import GPflow
import numpy as np

def plot(m):
    # plot model fit
    plt.figure()
    extra = (X.max() - X.min()) * 1.0
    Xtest = np.linspace(X.min() - extra, X.max() + extra, 300)[:, None]
    for i in range(20):
        s = m.predict_sampling(Xtest)
        plt.plot(Xtest, s, 'bo', mew=0, alpha=0.1)
    for i in range(3):
        s = m.predict_sampling_correlated(Xtest)
        plt.plot(Xtest, s, 'r', lw=1.2)
    plt.plot(X, Y, 'kx', mew=1.5, ms=8)

    for l in m.layers:
        Z = l.Z.value
        extra = (Z.max() - Z.min()) * 0.1
        Xtest = np.linspace(Z.min() - extra, Z.max() + extra, 100)[:, None]
        mu, var = l.predict_f(Xtest)
        plt.figure()
        plt.plot(Xtest, mu, 'r', lw=1.5)
        plt.plot(Xtest, mu - 2*np.sqrt(var), 'r--')
        plt.plot(Xtest, mu + 2*np.sqrt(var), 'r--')
        mu, var = l.predict_f(Z)
        plt.errorbar(Z.flatten(), mu.flatten(), yerr=2*np.sqrt(var).flatten(),
                     capsize=0, elinewidth=1.5, ecolor='r', linewidth=0)

def plot2(m):
    # plot model fit
    plt.figure()
    extra = (X.max() - X.min()) * 1.0
    Xtest = np.linspace(X.min() - extra, X.max() + extra, 300)[:, None]
    Ymu, Yvar = m.predict_f(Xtest)
    plt.plot(Xtest, Ymu, 'r', lw=1.5)
    plt.plot(Xtest, Ymu - 2 * np.sqrt(Yvar), 'r--')
    plt.plot(Xtest, Ymu + 2 * np.sqrt(Yvar), 'r--')
    plt.plot(X, Y, 'kx', mew=1.5, ms=8)

X = np.linspace(-3, 3, 100)[:, None]
Y = np.where(X < 0, -1, 1) + np.random.randn(100, 1) * 0.01
m = GPflow.coldeep.ColDeep(X, Y, (5,5,5), (15, 15,15,15))
for l in m.layers:
    l.Z.fixed = True
    l.kern.fixed = True
    #l.kern.lengthscales = 2.5
    l.beta.fixed = True
    l.q_mu = np.random.randn(*l.q_mu.shape)
m.layers[0].kern.lengthscales = 5.
m.optimize(maxiter=50, disp=1)
#m.optimize(tf.train.AdamOptimizer(0.01))
#plot(m)
plot2(m)
plt.show()

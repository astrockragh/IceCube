import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sympy import solve, Poly, Eq, Function, exp, lambdify, nonlinsolve, diff, Matrix
from sympy.abc import x, y

def dynamical(xdot, ydot, t_span=(0,100), x_span=(2,2), y_span=(2,2), res=50, solres=10):
    fxd=lambdify([x,y],xdot)
    fyd=lambdify([x,y],ydot)
    def fun(t,y):
        return np.array([fxd(y[0], y[1]), fyd(y[0],y[1])])

    def onclick(event):
        y0 = np.array([event.xdata,event.ydata])
        sol = solve_ivp(fun,t_span,y0,dense_output= True,vectorized=False)
        if not sol.success:
            ax.set_title('not successful in solving from given starting point')
            return
        path.set_data(sol.y)
        return

    #make base vector plot + solutions
    X = np.linspace(x_span[0],x_span[1],res)
    Y = np.linspace(y_span[0],y_span[1],res)
    xx, yy = np.meshgrid(X,Y)
    uv = fun(0,[xx,yy])

    fig, ax = plt.subplots()
    path, = ax.plot([0],[0])
    quiv = ax.quiver(xx,yy,uv[0]/abs(uv[0]),uv[1]/abs(uv[1]),alpha=0.5)
    fig.canvas.mpl_connect('button_press_event',onclick)
    #solution tracks
    r=solres# consider changing to keyword
    for x in X[::r]:
        for i,y in enumerate(Y[::r]):
            sol = solve_ivp(fun,t_span,np.array([x,y]),dense_output= True,vectorized=False)
            ax.plot(sol.y[0],sol.y[1],c='gray',alpha=0.3) # label=f'initial cond x:{x:.2f}, y:{y:.2f}'
# ax.legend()
    sols=nonlinsolve([xdot,ydot],[x,y])
    diff_fx=diff(xdot, x)
    diff_fy=diff(xdot, y)
    diff_gx=diff(ydot, x)
    diff_gy=diff(ydot, y)
    fp={}
    for j in range(len(sols.args)):
        xx, xy, yx, yy=diff_fx.evalf(subs={x:sols.args[j][0], y:sols.args[j][1]}),\
                        diff_fy.evalf(subs={x:sols.args[j][0], y:sols.args[j][1]}),\
                        diff_gx.evalf(subs={x:sols.args[j][0], y:sols.args[j][1]}),\
                        diff_gy.evalf(subs={x:sols.args[j][0], y:sols.args[j][1]})
        M=Matrix([[xx,xy],[yx,yy]])
        eigvec=np.array(M.eigenvects())
        fp[j]=np.array([sols.args[j][0], sols.args[j][1]]), eigvec[:,0],  \
        np.array([np.array(eigvec[:,2][0][0]),  np.array(eigvec[:,2][1][0])]), 
    for j in range(len(fp)):
        point, eigvals, eigvecs=fp[j]
        
        ax.scatter(point[0], point[1], label=stab(eigvals[0], eigvals[1]))
        ax.quiver(point[0], point[1], float(eigvecs[0][0][0]), float(eigvecs[0][1][0]), color='b', label=f'{eigvals[0]}')
        ax.quiver(point[0], point[1], float(eigvecs[1][0][0]), float(eigvecs[1][1][0]), color='r', label=f'{eigvals[1]}')


    ax.legend()    
    return fig, ax, fp
def stab(l1,l2):
    tau=l1+l2
    delta=l1*l2
    tau,delta=np.real(tau), np.real(delta)
    if delta<0:
        return 'Saddle node'
    if tau**2-4*delta==0 and delta!=0 and tau!=0:
        return 'Degenerate/star'
    if tau**2<4*delta and tau>0 and delta>0:
        return 'Unstable spiral'
    if tau**2<4*delta and tau<0 and delta>0:
        return 'Stable spiral'
    if delta>0 and tau==0:
        return 'Center'
    if tau**2>4*delta and tau>0 and delta>0:
        return 'Unstable node'
    if tau**2>4*delta and tau<0 and delta>0:
        return 'Stable node'
    if delta==0 and tau!=0:
        return 'Line of fixed points'
    if delta==0 and tau==0:
        return 'Plane of fixed points'
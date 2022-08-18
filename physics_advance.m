function v = physics_advance(v,rho)
% Advance physics-based model by one step

dt = 0.01;
sigma = 2*10;
beta = 2*8/3;

x = v(1);
y = v(2);
z = v(3);

eps = 0*[0.011 0.011 0.378]*1e-4;
epsx = eps(1);
epsy = eps(2);
epsz = eps(3);
noise_x = -epsx + 2*epsx*rand();
noise_y = -epsy + 2*epsy*rand();
noise_z = -epsz + 2*epsz*rand();

fx =@(x,y,z) sigma.*(y - x);
fy =@(x,y,z) x.*(rho - z) - y;
fz =@(x,y,z) x.*y - beta.*z;
k1x = dt.*fx(x,y,z);
k1y = dt.*fy(x,y,z);
k1z = dt.*fz(x,y,z);
k2x = dt.*fx(x+0.5*k1x,y+0.5*k1y,z+0.5*k1z);
k2y = dt.*fy(x+0.5*k1x,y+0.5*k1y,z+0.5*k1z);
k2z = dt.*fz(x+0.5*k1x,y+0.5*k1y,z+0.5*k1z);
k3x = dt.*fx(x+0.5*k2x,y+0.5*k2y,z+0.5*k2z);
k3y = dt.*fy(x+0.5*k2x,y+0.5*k2y,z+0.5*k2z);
k3z = dt.*fz(x+0.5*k2x,y+0.5*k2y,z+0.5*k2z);
k4x = dt.*fx(x+k3x,y+k3y,z+k3z);
k4y = dt.*fy(x+k3x,y+k3y,z+k3z);
k4z = dt.*fz(x+k3x,y+k3y,z+k3z);
x = x + k1x/6 + k2x/3 + k3x/3 + k4x/6 + noise_x;
y = y + k1y/6 + k2y/3 + k3y/3 + k4y/6 + noise_y;
z = z + k1z/6 + k2z/3 + k3z/3 + k4z/6 + noise_z;

v = [x; y; z];
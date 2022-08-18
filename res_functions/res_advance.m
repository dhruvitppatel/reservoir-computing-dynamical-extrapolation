function R = res_advance(A,Win,R,u,leak,bias)
% Iterate reservoir state forward by one step

R = (1-leak)*R + leak*tanh(A*R + Win*u + bias);
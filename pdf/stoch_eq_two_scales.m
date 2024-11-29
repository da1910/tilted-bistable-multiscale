function result = stoch_eq_two_scales(alp, eta)
    % Solve stochastic mulstiscale equation 
    % Euler explicit time scheme is used
    
    eps = 0.1; tpe = 2 * pi / eps;
    dt  = 0.01 * eps^2;
    T  = 4000;
    steps = round(T / dt);
    
    ndt = 100;  % sampling time
    t_init = 1000;  % collect data starting time
    n_init = round(t_init / dt);
    Time = (1:(steps - n_init) / ndt) * dt * ndt;
    
    samples = 40;  % different noise realizations
    
    u = zeros(numel(Time), samples);
    
    % Coefficients
    sig = 0.1;
    del = 10;
    
    gd  = @(x)(exp(-x.^2/del^2));
    V0  = @(x)(x.^4/4 - alp*x.^2/2 + eta*x);
    dV0 = @(x)(x.^3 - alp*x + eta);
    V1  = @(x)(-(1/2)*gd(x).*(x.^2).*sin(tpe*x));
    dV1 = @(x)(gd(x).*(-x.*sin(tpe*x) + (x.^3/del^2).*sin(tpe*x) - (pi*x.^2/eps).*cos(tpe*x)));
    
    % Potential 2 scales
    V  = @(x)(V0(x) + V1(x));
    dV = @(x)(dV0(x) + dV1(x));
     
    %figure;
    for j=1:samples 
        j
        u0 = 0; % initial condition   
        ic = 0;
        for i=1 : steps       
            k1 = -dV(u0);    
            dW = sqrt(dt)*randn(1, 1);  % noise generated at each time step
            b  = (sqrt(2 * sig));
            u0 = u0 + k1 * dt + b * dW;
          
            if (mod(i, ndt) == 0) && (i > n_init)  % time sampling
                ic = ic + 1;
                u(ic, j) = u0;                      
            end
        end    
    end
    
    % Histogram of sde data
    u_data = reshape(u, 1, []);
    
    bin = [0.01 0.1];
    figure;
    for n = 1:numel(bin)
        nb  = round((max(u_data) - min(u_data)) / bin(n)); 
        [counts, edges] = histcounts(u_data, nb);
    
        x = (edges(1:end - 1) + edges(2:end)) / 2;
        pdf = counts / (numel(u_data) * bin(n));
        plot(x, pdf);
        hold all
    end
    
    result = struct("time", Time, "uData", u_data, "samples", samples, "sigma", sig, "alpha", alp, "eta", eta);
end
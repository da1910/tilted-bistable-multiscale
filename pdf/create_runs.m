alphas = [0.5, -0.5, -0.25];
etas = [0, 0.1];
numberOfRuns = length(alphas) * length(etas);

for i = 1:length(alphas)
    alpha = alphas(i);
    for j = 1:length(etas)
        eta = etas(j);
        index = (i - 1) * length(etas) + j;
        sprintf("Running with alpha=%f and eta=%f", alpha, eta)
        result(numberOfRuns + 1 - index) = stoch_eq_two_scales(alpha, eta);
    end
end

save("two_scale_data_wide.mat", "result")
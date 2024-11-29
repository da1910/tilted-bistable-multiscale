jsonObjects = struct("alpha", {}, "beta", {}, "eta", {}, "bin_widths", {}, "bin_edges", {}, "counts", {});

for entry = result
    bin = [0.01 0.1];

    for n = 1:numel(bin)
        nb  = round((max(entry.uData) - min(entry.uData)) / bin(n)); 
        [counts, edges] = histcounts(entry.uData, nb);
    
        x = (edges(1:end - 1) + edges(2:end)) / 2;
        pdf = counts / (numel(entry.uData) * bin(n));

        jsonObjects(end+1) = struct("alpha", entry.alpha, "beta", 1/entry.sigma, "eta", entry.eta, "bin_widths", bin(n), "bin_edges", edges, "counts", counts);
    end
end

data = jsonencode(jsonObjects);
fid = fopen("pdf_data.json", 'w');
fprintf(fid,'%s',data);
fclose(fid);
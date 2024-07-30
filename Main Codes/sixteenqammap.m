function symbols = sixteenqammap(bits)
  
    symbols = zeros(1, numel(bits)/4);
   
    k =1;
    for i = 1:4:numel(bits)
        if bits(i) == '0' && bits(i+1) == '0'
            bc = -3;
        elseif bits(i) == '0' && bits(i+1) == '1'
            bc = -1;
        elseif bits(i) == '1'&& bits(i+1) == '1'
            bc = 1;
        elseif bits(i) == '1'&& bits(i+1) == '0'
            bc = 3;
        end
        
        if bits(i+2) == '0' && bits(i+3) == '0'
            bs = -3;
        elseif bits(i+2) == '0' && bits(i+3) == '1'
            bs = -1;
        elseif bits(i+2) == '1'&& bits(i+3) == '1'
            bs = 1;
        elseif bits(i+2) == '1'&& bits(i+3) == '0'
            bs = 3;
        end
        
        symbols(k) = bc + 1i * bs;
        k = k+1;
    end
end
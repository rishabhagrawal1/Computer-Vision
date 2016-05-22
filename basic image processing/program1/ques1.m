function [guass] = fun1(w, s)
    [h,w] = size(I);
    B = zeros(w,w);
    k = (w-1)/2;
    summ =0;
    for(i = 1 : w)
        for(j = 1 : w)
            B(i,j) = (1/(2*pi*power(s,2)))*exp(-1* (power(i-k-1,2) + power(j-k-1,2)) / (2 * power(s,2)));
            summ = summ + (B(i,j));
        end
    end

    B = B/summ;
    guass = B;
end
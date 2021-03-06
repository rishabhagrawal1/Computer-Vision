%function to provide laplacian of gaussian%
function [LC] = ques4()
    LK1 = [0  1  0 
           1 -4  1
           0  1  0];
    LK2 = [1  1  1 
           1 -8  1
           1  1  1];  
    
    G = fun1(11,1)
    LC = conv2(G , LK1);
end

%function to give gaussian values%
function [guass] = fun1(w, s)
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
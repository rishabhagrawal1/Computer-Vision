%function to detect edge in give image using laplacian of gaussian%
function [LGFI] = lapGausFilterImage(I, t)
    [h,w] = size(I);    
    %use old functions to get the gaussian and convolution with laplacian discrete approximation%
    LG = conv2(I , lapGaus());
    %create new matrix for output%
    LGFI = zeros(h,w);
    
    for(i = 1 : h)
        for(j = 1 : w)
            if((i > 1 && j > 1) && LG(i,j) < 0 && LG(i-1,j-1) > 0)
                if(abs(LG(i,j) - LG(i-1,j-1)) > t)
                    LGFI(i,j) = 1;
                end
            elseif((i > 1) && LG(i,j) < 0 && LG(i-1,j) > 0)
                if(abs(LG(i,j) - LG(i-1,j)) > t)
                    LGFI(i,j) = 1;
                end
            elseif((j > 1) && LG(i,j) < 0 && LG(i,j-1) > 0)
                if(abs(LG(i,j) - LG(i,j-1)) > t)
                    LGFI(i,j) = 1;
                end
            elseif((i < h) && LG(i,j) < 0 && LG(i+1,j) > 0)
                if(abs(LG(i,j) - LG(i+1,j)) > t)
                    LGFI(i,j) = 1;
                end
            elseif((j < w) && LG(i,j) < 0 && LG(i,j+1) > 0)
                if(abs(LG(i,j) - LG(i+1,j)) > t)
                    LGFI(i,j) = 1;
                end
            elseif((i < h && j < w) && LG(i,j) < 0 && LG(i+1,j+1) > 0)    
                if(abs(LG(i,j) - LG(i+1,j)) > t)
                    LGFI(i,j) = 1;
                end
            end    
        end
    end
end

%function to provide laplacian of gaussian%
function [LC] = lapGaus()
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
function funs = ques2
  funs.gausx=@gaussian_x;
  funs.gausy=@gaussian_y;
  funs.gausxy=@gaussian_xy;
  funs.comp=@compare_images;
end

function [gaussx] = gaussian_x(w, s)
B = zeros(1,w);
k = (w-1)/2;
summ =0;

for(i = 1 : w)
    B(1,i) = (1/(2*pi*power(s,2)))*exp(-1* power(i-k-1,2) / (2 * power(s,2)));
    summ = summ + (B(1,i));
end
gaussx = B/summ;
end


function [gaussy] = gaussian_y(w, s)
B = zeros(w,1);
k = (w-1)/2;
summ =0;

for(i = 1 : w)
   B(i,1) = (1/(2*pi*power(s,2)))*exp(-1* power(i-k-1,2) / (2 * power(s,2)));
   summ = summ + (B(i,1));
end
gaussy = B/summ;
end

function [guass] = gaussian_xy(w, s)
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

function [max_diff] = compare_images(I, J)
    [h,w] = size(I);
    max_diff = 0;
    for(i = 1 : h)
        for(j = 1 : w)
            if( abs(I(i,j) - J(i,j)) > max_diff)
                max_diff = abs(I(i,j) - J(i,j))
            end
        end
    end
end


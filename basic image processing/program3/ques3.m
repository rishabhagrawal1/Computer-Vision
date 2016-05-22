function [G] = sobel(I, t)
    [h,w] = size(I);

    SX =[-1 0 1
         -2 0 2
         -1 0 1]

    SY =[-1 -2 -1
         -2  0  2
          1  2  1]

    GX = conv2(I,SX,'same');
    GY = conv2(I,SY,'same');
    
    for(i = 1: h)
        for(j = 1: w)
            G(i,j) = sqrt(power(GX(i,j),2) + power(GY(i,j),2));
        end
    end
    
    % applying threshold %
    if(t > 0)
        for(i = 1 : h)
            for(j = 1 : w)
                if(G(i,j) > t)
                    G(i,j) = 1;
                else
                    G(i,j) = 0;
                end
            end
        end
    end
end

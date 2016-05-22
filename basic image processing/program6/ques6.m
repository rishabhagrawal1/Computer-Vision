% function to implement canny enahancer %
function [CEI] = canny_enhancer(I,t)
    [h,w] = size(I);
    
    % applying Gaussian on image %
    G = guas(5,3);
    GI = conv2(I,G,'same');
    [SI, ST] = sobel(GI,0);
    % now apply non max suppression and threshold after that%
    CEI = nonmax_supression(SI, ST, t);
end

% function to implement non max supression %
function [NSI] = nonmax_supression(SI, ST, t)
    [h,w] = size(SI);
    NSI = zeros(h,w);
    
    for(i = 1 : h)
        for(j = 1 : w)
            if((ST(i,j) < 22.5 && ST(i,j) >= 0) || (ST(i,j) > -22.5 && ST(i,j) < 0))
                ST(i,j) = 0;
            elseif(ST(i,j) >= 22.5 && ST(i,j) < 67.5)
                ST(i,j) = 45;
            elseif(ST(i,j) > -67.5 && ST(i,j) <= -22.5)
                ST(i,j) = 135;
            elseif((ST(i,j) > -90 && ST(i,j) <= -67.5 ) || (ST(i,j) >= 67.5 && ST(i,j) < 90))
                ST(i,j) = 90;
            end
        end
    end
    NSI = calculate_pixel_values(SI, ST);
    
    % applying threshold %
    if(t > 0)
        for(i = 1 : h)
            for(j = 1 : w)
                if(NSI(i,j) > t)
                    NSI(i,j) = 1;
                else
                    NSI(i,j) = 0;
                end
            end
        end
    end
end

function [SI] = calculate_pixel_values(SI, ST)
    [h,w] = size(SI);
    TEMP = zeros(h, w);
    for(i = 2 : h-1)
        for(j = 2 : w-1)
            if (ST(i,j) == 0)
                if (SI(i, j) < SI(i-1, j) || SI(i, j) < mod(i+1, j))
                    SI(i,j) = 0;
                end 
            end
            if (ST(i,j) == 45)
                if (SI(i, j) < SI(i+1, j-1) || SI(i, j) < SI(i-1, j+1))
                    SI(i,j) = 0;
                end
            end
            if (ST(i,j) == 90)
                if (SI(i, j) < SI(i, j-1) || SI(i, j) < SI(i, j+1))
                    SI(i,j) = 0;
                end
            end
            if (ST(i,j) == 135)
                if (SI(i, j) < SI(i-1, j-1) || SI(i, j) < SI(i+1, j+1))
                    SI(i,j) = 0;
                end 
            end
        end
    end    
end

%function to apply Sobel%
function [G,GTHETHA] = sobel(I, t)
    [h,w] = size(I);
    
    %to store gradient%
    GTETHA = zeros(h,w);
    
    SX =[-1 0 1
         -2 0 2
         -1 0 1];

    SY =[-1 -2 -1
         -2  0  2
          1  2  1];

    %to store the x and y gradient components%  
    GX = conv2(I,SX,'same');
    GY = conv2(I,SY,'same');
    
    for(i = 1: h)
        for(j = 1: w)
            G(i,j) = sqrt(power(GX(i,j),2) + power(GY(i,j),2));
            if(GY(i,j) == 0)
                GTHETHA(i,j) = 0;
            elseif(GY(i,j) == 0)    
                GTHETHA(i,j) = 1000;
            else
                GTHETHA(i,j) = atand(GY(i,j)/GX(i,j));
            end
        end
    end
end


% function to give gaussian filter %
function [guass] = guas(w, s)
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
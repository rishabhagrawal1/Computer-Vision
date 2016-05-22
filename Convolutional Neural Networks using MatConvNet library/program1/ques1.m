%cd D:\vision\assign_3\matconvnet-master
addpath matlab
run('D:\vision\vlfeat-0.9.20\toolbox\vl_setup')
addpath D:\vision\assign_3\cifar-10-batches-mat
% Read an example image
x = imread('peppers.png') ;
setup;
% Convert to single format
x = im2single(x) ;

% Visualize the input x
%figure(1) ; clf ; imagesc(x)
stride = 1
pad = 0
wx = [-1 0 1 
      -2 0 2 
      -1 0 1 ] ;
  
wy = [-1 -2 -1 
       0  0  0 
       1  2  1 ] ;
   
wx = single(repmat(wx, [1, 1, 3])) ;
wy = single(repmat(wy, [1, 1, 3])) ;

%%Stacking both  x and y dimensional filters
w(:,:,:,1) = wx;
w(:,:,:,2) = wy;

lap1 = vl_nnconv(x, w, [], 'stride', stride, 'pad', pad) ;
figure(3) ; clf ; vl_imarraysc(lap1) ;colormap gray ;

z = vl_nnrelu(lap1) ;
figure(4) ; clf ; vl_imarraysc(z) ;colormap gray ;

pool = vl_nnpool(z, 15) ;
figure(5) ; clf ; vl_imarraysc(pool) ; colormap gray ;
clear;
%% Symmetric Class%%
Force_class_1 = zeros(150,16);

f_high = 45.00;
f_mid = 29;
f_low = 40;

sd_high = 5.0;
sd_mid = 18.0;
sd_low = 5.0;
sd_zero = 15.0;

data_high = abs(f_high+sd_high.*randn(150,16));
% data_low = abs(f_low+sd_low.*randn(75,8));




% Force_class_1(:,[2,3,6,7,10,11,14,15])=data_high;
% Force_class_1(:,[1,4,5,8,9,12,13,16])=data_low;
Force_class_1(:,:)=data_high;

class1 = Force_class_1;


%% Asymmetric Class%%
Force_class_2 = zeros(75,16);
Force_class_3 = zeros(75,16);


f_high = 60;
f_mid = 35;
f_low = 3;

sd_high = 10.0;
sd_mid = 18.0;
sd_low = 0.1;
sd_zero = 15.0;

data_high = abs(f_high+sd_high.*randn(75,12));
data_low = abs(f_low+sd_low.*randn(75,4));



Force_class_2(:,[1,2,5,6])=data_low;
Force_class_2(:,[3,4,7,8,9,10,11,12,13,14,15,16])=data_high;



data_high_ = abs(f_high+sd_high.*randn(75,12));
data_low_ = abs(f_low+sd_low.*randn(75,4));





Force_class_3(:,[9,10,13,14])=data_low_;
Force_class_3(:,[1,2,3,4,5,6,7,8,11,12,15,16])=data_high_;

class2 = [Force_class_2(1:38,:);   Force_class_3(1:37,:)];

%%Surface Plotting class samples%%


x_axis = 0:3;
y_axis = 0:3;
[surfx,surfy] = meshgrid(x_axis,y_axis);

samples = [1,2,3,4,5,6,7,8,9];

figure;
title('Symmetric Class')
for i=1:9
    subplot(3,3,i);
    T = flip(reshape(Force_class_1(samples(i),:),[4,4]));
    surf(surfx,surfy,T,'FaceAlpha',0.9,'edgecolor', 'none');
    xlabel('x'); ylabel('y'); zlabel('F');
end
colormap();

figure;
for i=1:9
    subplot(3,3,i);
    T = flip(reshape(Force_class_2(samples(i),:),[4,4]));
    surf(surfx,surfy,T,'FaceAlpha',0.9,'edgecolor', 'none');
    xlabel('x'); ylabel('y'); zlabel('F');
end
colormap();

figure;
for i=1:9
    subplot(3,3,i);
    T = flip(reshape(Force_class_3(samples(i),:),[4,4]));
    surf(surfx,surfy,T,'FaceAlpha',0.9,'edgecolor', 'none');
    xlabel('x'); ylabel('y'); zlabel('F');
end
colormap();



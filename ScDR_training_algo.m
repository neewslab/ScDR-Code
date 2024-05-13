clear; 

%% Variable initialization  %%

num_runs = 1;


epochs = 20;
del_w = 0.1;

num_feat = 16;

num_spr = 40;

n_fold = 5;

iteration = 1;


%% Load Dataset  %%

c1=struct2cell(load('C:\Users\duttahr1\OneDrive - Michigan State University\PBL_Plots\Posture_Dataset\Data_Leg_Crossed\class1_1.mat'));
c2=struct2cell(load('C:\Users\duttahr1\OneDrive - Michigan State University\PBL_Plots\Posture_Dataset\Data_Leg_Crossed\class2_1.mat'));

class_1_data = c1{1};
class_2_data = c2{1};


%% Train-test Split  %%


trn_size  = length(class_1_data)*(n_fold-1)/n_fold;
test_size = length(class_1_data)/n_fold;


%% Define Learning hyperparameters, such as learning rate etc.  %%

lr_max = 150;   % 150 for x 6 and 10
lr_dec = 250;
% lr_dec = 500;

batch_size =1;



k_r1 = 2.0;
k_r2 = 2.0;
k_min = 1;
k_max = 10;



all_runs = zeros(num_runs,epochs);

all_runs_val = zeros(num_runs,epochs);


%% Loop for multiple runs of learning for statistical confidence of performance values  %%

for run =1:num_runs



acc_acr_k = zeros(n_fold,epochs,7);
tr_acc_acr_k = zeros(n_fold,epochs,7);


%% Initialize weight matrices  %%

weight_evolution1 = zeros(epochs,num_spr);
weight_evolution2 = zeros(epochs,num_spr);
weight_evolution3 = zeros(epochs,num_spr);
weight_evolution4 = zeros(epochs,num_spr);
weight_evolution5 = zeros(epochs,num_spr);




w_0_list = zeros(n_fold,num_spr);
data_list = zeros(n_fold,num_feat);




%% Loop for k-fold validation  %%

for k = 1:n_fold
     
    
    stop_crt = 0;
    stop_crt_cnt = 0;
    
    test_set_c1 = class_1_data((k-1)*test_size+1:k*test_size,:);
    test_set_c2 = class_2_data((k-1)*test_size+1:k*test_size,:);
    
    
    trn_set_c1 = class_1_data;
    trn_set_c2 = class_2_data;
    
    trn_set_c1((k-1)*test_size+1:k*test_size,:)=[];
    trn_set_c2((k-1)*test_size+1:k*test_size,:)=[];
    
    
    train_mean_c1 = mean(trn_set_c1);
    train_mean_c2 = mean(trn_set_c2);
    
    disp_matrix = zeros(epochs,iteration);


    loss_list = [];

    w = 5.5*ones(1,num_spr);
    
    w_0_list(k,:)=w;


    accuracy = [];
    tr_acc =[];
    
    thres_list = [];
    
    
    pos_cls = 1;
    
    
    op_c1 = [];
    op_c2 = [];
    

    sd_mat = [1,1];


    %% Loop for ScDR learning epochs  %%

    for e = 1:epochs

               
        
        lr = lr_max*exp(-e/lr_dec);

        
        
        target_c1 = calculate_disp(w, train_mean_c1, num_feat);
        target_c2 = calculate_disp(w, train_mean_c2, num_feat);
        
        
        
       
        %% Loop for iteration for changing target, default value of iteration is 1, meaning no loop%%

        for iter = 1:iteration


            %% Weight matrix updates to check direction of loss minimization%%
            if num_spr == 40
                wt_change_pos = [w(1)+del_w,w(2:end)
                            w(1),w(2)+del_w,w(3:end)
                            w(1:2),w(3)+del_w,w(4:end)
                            w(1:3),w(4)+del_w,w(5:end)
                            w(1:4),w(5)+del_w,w(6:end)
                            w(1:5),w(6)+del_w,w(7:end)
                            w(1:6),w(7)+del_w,w(8:end)
                            w(1:7),w(8)+del_w,w(9:end)
                            w(1:8),w(9)+del_w,w(10:end)
                            w(1:9),w(10)+del_w,w(11:end)
                            w(1:10),w(11)+del_w,w(12:end)
                            w(1:11),w(12)+del_w,w(13:end)
                            w(1:12),w(13)+del_w,w(14:end)
                            w(1:13),w(14)+del_w,w(15:end)
                            w(1:14),w(15)+del_w,w(16:end)
                            w(1:15),w(16)+del_w,w(17:end)
                            w(1:16),w(17)+del_w,w(18:end)
                            w(1:17),w(18)+del_w,w(19:end)
                            w(1:18),w(19)+del_w,w(20:end)
                            w(1:19),w(20)+del_w,w(21:end)
                            w(1:20),w(21)+del_w,w(22:end)
                            w(1:21),w(22)+del_w,w(23:end)
                            w(1:22),w(23)+del_w,w(24:end)
                            w(1:23),w(24)+del_w,w(25:end)
                            w(1:24),w(25)+del_w,w(26:end)
                            w(1:25),w(26)+del_w,w(27:end)
                            w(1:26),w(27)+del_w,w(28:end)
                            w(1:27),w(28)+del_w,w(29:end)
                            w(1:28),w(29)+del_w,w(30:end)
                            w(1:29),w(30)+del_w,w(31:end)
                            w(1:30),w(31)+del_w,w(32:end)
                            w(1:31),w(32)+del_w,w(33:end)
                            w(1:32),w(33)+del_w,w(34:end)
                            w(1:33),w(34)+del_w,w(35:end)
                            w(1:34),w(35)+del_w,w(36:end)
                            w(1:35),w(36)+del_w,w(37:end)
                            w(1:36),w(37)+del_w,w(38:end)
                            w(1:37),w(38)+del_w,w(39:end)
                            w(1:38),w(39)+del_w,w(40:end)
                            w(1:39),w(40)+del_w];
                        
                        
            wt_change_neg = [w(1)-del_w,w(2:end)
                            w(1),w(2)-del_w,w(3:end)
                            w(1:2),w(3)-del_w,w(4:end)
                            w(1:3),w(4)-del_w,w(5:end)
                            w(1:4),w(5)-del_w,w(6:end)
                            w(1:5),w(6)-del_w,w(7:end)
                            w(1:6),w(7)-del_w,w(8:end)
                            w(1:7),w(8)-del_w,w(9:end)
                            w(1:8),w(9)-del_w,w(10:end)
                            w(1:9),w(10)-del_w,w(11:end)
                            w(1:10),w(11)-del_w,w(12:end)
                            w(1:11),w(12)-del_w,w(13:end)
                            w(1:12),w(13)-del_w,w(14:end)
                            w(1:13),w(14)-del_w,w(15:end)
                            w(1:14),w(15)-del_w,w(16:end)
                            w(1:15),w(16)-del_w,w(17:end)
                            w(1:16),w(17)-del_w,w(18:end)
                            w(1:17),w(18)-del_w,w(19:end)
                            w(1:18),w(19)-del_w,w(20:end)
                            w(1:19),w(20)-del_w,w(21:end)
                            w(1:20),w(21)-del_w,w(22:end)
                            w(1:21),w(22)-del_w,w(23:end)
                            w(1:22),w(23)-del_w,w(24:end)
                            w(1:23),w(24)-del_w,w(25:end)
                            w(1:24),w(25)-del_w,w(26:end)
                            w(1:25),w(26)-del_w,w(27:end)
                            w(1:26),w(27)-del_w,w(28:end)
                            w(1:27),w(28)-del_w,w(29:end)
                            w(1:28),w(29)-del_w,w(30:end)
                            w(1:29),w(30)-del_w,w(31:end)
                            w(1:30),w(31)-del_w,w(32:end)
                            w(1:31),w(32)-del_w,w(33:end)
                            w(1:32),w(33)-del_w,w(34:end)
                            w(1:33),w(34)-del_w,w(35:end)
                            w(1:34),w(35)-del_w,w(36:end)
                            w(1:35),w(36)-del_w,w(37:end)
                            w(1:36),w(37)-del_w,w(38:end)
                            w(1:37),w(38)-del_w,w(39:end)
                            w(1:38),w(39)-del_w,w(40:end)
                            w(1:39),w(40)-del_w];
            end

            
            


            w_int = w;
            
            
            %% Find the direction of loss minimization %%
            
            for x_itr = 1:num_spr
                
                temp = wt_change_pos(x_itr,:);
            
                tr_cnt = 0;
                
                
                for t = 1:trn_size
                    test_class = trn_set_c2;
                    tr = target_c2;
                    grd_th = 2;

                    test_ex = test_class(t,:);

                    test_out = calculate_disp(temp, test_ex, num_feat);

                    decision_vec = [norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;

                    tst_cls = find(decision_vec == min(decision_vec));



                    if grd_th == tst_cls
                        tr_cnt = tr_cnt+1;
                    end


                end

                for t = 1:trn_size
                    test_class = trn_set_c1;
                    tr = target_c1;
                    grd_th = 1;

                    test_ex = test_class(t,:);

                    test_out = calculate_disp(temp, test_ex, num_feat);

                    decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;

                    tst_cls = find(decision_vec == min(decision_vec));


                    if grd_th == tst_cls
                        tr_cnt = tr_cnt+1;
                    end


                end


                
                
                
                error_pos = (2*trn_size-tr_cnt)/(2*trn_size);
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                temp = wt_change_neg(x_itr,:);
            
                tr_cnt = 0;
                
                
                for t = 1:trn_size
                    test_class = trn_set_c2;
                    tr = target_c2;
                    grd_th = 2;


                    test_ex = test_class(t,:);

                    test_out = calculate_disp(temp, test_ex, num_feat);

                    decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;

                    tst_cls = find(decision_vec == min(decision_vec));



                    if grd_th == tst_cls
                        tr_cnt = tr_cnt+1;
                    end


                end

                for t = 1:trn_size
                    test_class = trn_set_c1;
                    tr = target_c1;
                    grd_th = 1;

                    test_ex = test_class(t,:);

                    test_out = calculate_disp(temp, test_ex, num_feat);

                    decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;

                    tst_cls = find(decision_vec == min(decision_vec));


                    if grd_th == tst_cls
                        tr_cnt = tr_cnt+1;
                    end


                end


                
                
                
                error_neg = (2*trn_size-tr_cnt)/(2*trn_size);
                
                
                
                
                
                
                
                
                
                
                
                
                
                temp = w;
            
                tr_cnt = 0;
                
                
                for t = 1:trn_size
                    test_class = trn_set_c2;
                    tr = target_c2;
                    grd_th = 2;

                    test_ex = test_class(t,:);

                    test_out = calculate_disp(temp, test_ex, num_feat);

                    decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;

                    tst_cls = find(decision_vec == min(decision_vec));



                    if grd_th == tst_cls
                        tr_cnt = tr_cnt+1;
                    end


                end

                for t = 1:trn_size
                    test_class = trn_set_c1;
                    tr = target_c1;
                    grd_th = 1;

                    test_ex = test_class(t,:);

                    test_out = calculate_disp(temp, test_ex, num_feat);

                    decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;

                    tst_cls = find(decision_vec == min(decision_vec));


                    if grd_th == tst_cls
                        tr_cnt = tr_cnt+1;
                    end


                end

                
                error_curr = (2*trn_size-tr_cnt)/(2*trn_size);
                
                
                
                err = [error_curr,error_pos,error_neg];
                
                
                %% Rule for weight update based on minimum loss direction %%
                
                
                if find(err==min(err))==2
                    w_int(x_itr)= min(w(x_itr)+lr*abs((error_pos-error_curr)/error_curr),k_max);
                elseif find(err==min(err))==3
                    w_int(x_itr)= max(w(x_itr)-lr*abs((error_neg-error_curr)/error_curr),k_min);
                elseif error_pos == error_neg
                    if error_pos<error_curr
                        w_int(x_itr)= min(w(x_itr)+lr*abs((error_pos-error_curr)/error_curr),k_max);
                    end
                end
            
                
                w = w_int;
                                
            end
                        
            
           %% Store weight matrices for further processing %%

            
            if k ==1
                weight_evolution1(e,:) = w;
            end

            if k ==2
                weight_evolution2(e,:) = w;
            end

            if k ==3
                weight_evolution3(e,:) = w;
            end

            if k ==4
                weight_evolution4(e,:) = w;
            end

            if k ==5
                weight_evolution5(e,:) = w;
            end
            
            

        end





        %% Finding Test accuracy    %%

        

        
        
        cnt = 0;
        cls_1_tp=0;
        cls_2_tp=0;
        
        fp12 = 0;
        
        fp21 = 0;
        
        

        cls_1_data_cnt = 0;
        cls_2_data_cnt = 0;

        for t = 1:test_size
            test_class = test_set_c2;
            tr = target_c2;
            grd_th = 2;
            cls_2_data_cnt=cls_2_data_cnt+1;
            



            test_ex = test_class(t,:);
            
            test_out = calculate_disp(w, test_ex, num_feat);

            
            decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;
            
            tst_cls = find(decision_vec == min(decision_vec));
            

%             if min([norm(test_out-target_c1), norm(test_out-target_c2), norm(test_out-target_c3)])
%                 tst_cls = pos_cls;
%             else
%                 tst_cls = 3-pos_cls;
%             end


            if grd_th == tst_cls
                cnt = cnt+1;
                cls_2_tp =  cls_2_tp+1;
            elseif tst_cls == 1
                fp21 = fp21+1;
            end

%             perf = [test_out,label,tr];
% 
%             performance(e,t,:)=perf;

        end
        
        
        
        
        for t = 1:test_size
            test_class = test_set_c1;
            label=-1.0;
            tr = target_c1;
            grd_th = 1;
            cls_1_data_cnt=cls_1_data_cnt+1;
            



            test_ex = test_class(t,:);

            test_out = calculate_disp(w, test_ex, num_feat);
            
            
            decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;
            
            tst_cls = find(decision_vec == min(decision_vec));


            if grd_th == tst_cls
                cnt = cnt+1;
                cls_1_tp =  cls_1_tp+1;
            elseif tst_cls == 2
                fp12 = fp12+1;
            end

%             perf = [test_out,label,tr];
% 
%             performance(e,t,:)=perf;

        end
        
        
        
        
        
        
        

        
        
    %% Finding Training accuracy   %%
    
    tr_cnt = 0;
    tr_cls_1_tp=0;
    tr_cls_2_tp=0;
    
    tr_cls_1_data_cnt = 0;
    tr_cls_2_data_cnt = 0;
    
    trn_fp12 = 0;
    trn_fp21 = 0;
    
    
    for t = 1:trn_size
        test_class = trn_set_c2;
        tr = target_c2;
        grd_th = 2;
        tr_cls_2_data_cnt=tr_cls_2_data_cnt+1;
        

        test_ex = test_class(t,:);
        
        test_out = calculate_disp(w, test_ex, num_feat);
        
        decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;
            
        tst_cls = find(decision_vec == min(decision_vec));
        
        
        
        
        
        if grd_th == tst_cls
            tr_cnt = tr_cnt+1;
            if tst_cls == 1
                tr_cls_1_tp =  tr_cls_1_tp+1;
            elseif tst_cls == 2
                tr_cls_2_tp =  tr_cls_2_tp+1;
            end
        elseif tst_cls == 1
            trn_fp21 = trn_fp21+1;
        end

    end
    
    for t = 1:trn_size
        test_class = trn_set_c1;
        tr = target_c1;
        grd_th = 1;
        tr_cls_1_data_cnt=tr_cls_1_data_cnt+1;

        test_ex = test_class(t,:);
        
        test_out = calculate_disp(w, test_ex, num_feat);
        
        decision_vec =[norm(test_out-target_c1), norm(test_out-target_c2)]./sd_mat;
            
        tst_cls = find(decision_vec == min(decision_vec));
        
        
        if grd_th == tst_cls
            tr_cnt = tr_cnt+1;
            if tst_cls == 1
                tr_cls_1_tp =  tr_cls_1_tp+1;
            elseif tst_cls == 2
                tr_cls_2_tp =  tr_cls_2_tp+1;
            end
        elseif tst_cls == 2
            trn_fp12 = trn_fp12+1;
        end

        
    end
    
    
    
        %% Store accuracy on entire dataset and also training accuracy %%


        accuracy =[accuracy;[cnt,cls_1_tp,cls_2_tp, fp12,fp21, cls_1_data_cnt,cls_2_data_cnt]];
        
        tr_acc =[tr_acc;[tr_cnt,tr_cls_1_tp,tr_cls_2_tp,trn_fp12,trn_fp21, tr_cls_1_data_cnt,tr_cls_2_data_cnt]];

        
        


%         disp(e*100/epochs);
    

    end
    
    
    
    acc_acr_k(k,:,:) = accuracy;
    
    tr_acc_acr_k(k,:,:) = tr_acc;
    
    
    disp(['Run ID:',num2str(run),'Fold ID',num2str(k)]);
end

%% Find accuracy over different training data folds %%

mean_acc=mean(acc_acr_k,1);

mean_tr_acc=mean(tr_acc_acr_k,1);

mean_tot_acc = mean_acc+mean_tr_acc;


acc_run = (mean_tr_acc(1,:,1)+mean_acc(1,:,1))./(2*(test_size+trn_size));




all_runs(run,:)= acc_run;
all_runs_val(run,:)= mean_acc(1,:,1)./(2*(test_size));


% disp(run);
end



%% Plots for perfromance %%


figure;
plot(all_runs','k','linewidth',1.2); hold on;
% ylim([0.9995 1.0003]);
xlabel('Learning Epochs'); ylabel('Accuracy'); grid on;
title('Average Accuracy (Entire Dataset)');




mmmt(:,:) = mean_tot_acc(1,:,:);
mmm(:,:) = mean_tr_acc(1,:,:);
m_a(:,:) = mean_acc(1,:,:);




disp(['Accuracy (Entire dataset): ',num2str(mean(mean(all_runs(:,epochs-3:epochs))))]);




%% Function for computing displacement from a given force matrix %%


function disp_vec = calculate_disp(k_mat, F, nf)


    id = [0  1  3  5  7  0   9  11  13  15  0   17  19  21 23   0   25   27   29   31   0   1   9   17   25   0   3   11   19   27   0   5   13   21   29   0   7   15   23   31; ...
      0  2  4  6  8  0  10  12  14  16  0   18  20  22 24   0   26   28   30   32   0   2   10  18   26   0   4   12   20   28   0   6   14   22   30   0   8   16   24   32; ...
      1  3  5  7  0  9  11  13  15   0  17  19  21  23  0   25  27   29   31   0    1   9   17  25    0   3   11  19   27    0   5  13   21   29   0    7  15   23   31   0; ...
      2  4  6  8  0  10 12  14  16   0  18  20  22  24  0   26  28   30   32   0    2   10  18  26    0   4   12  20   28    0   6  14   22   30   0    8  16   24   32   0];


    theta_h = 0;
    theta_v = 0;
    
    
    F_h = F.*cos(theta_h*pi/180);
    F_v = F.*cos(theta_v*pi/180);
    
%     F_1h = F(1)*cos(theta_h*pi/180);
%     F_2h = F(2)*cos(theta_h*pi/180);
%     F_3h = F(3)*cos(theta_h*pi/180);
%     F_4h = F(4)*cos(theta_h*pi/180);
%     
%     
%     F_1v = F(1)*cos(theta_v*pi/180);
%     F_2v = F(2)*cos(theta_v*pi/180);
%     F_3v = F(3)*cos(theta_v*pi/180);
%     F_4v = F(4)*cos(theta_v*pi/180);


%     k_matrix = reshape(k_mat,8,5);
% 
%     x_hor = zeros(4,4);
%     x_ver = zeros(4,4);

     
    if nf == 16
        
        
        % Form element stiffness matrices
        
        [k1,ft1] = truss2d(10,0,10,10,k_mat(1),1);
        [k2,ft2] = truss2d(10,10,10,20,k_mat(2),1);
        [k3,ft3] = truss2d(10,20,10,30,k_mat(3),1);
        [k4,ft4] = truss2d(10,30,10,40,k_mat(4),1);
        [k5,ft5] = truss2d(10,40,10,50,k_mat(5),1);
        [k6,ft6] = truss2d(20,0,20,10,k_mat(6),1);
        [k7,ft7] = truss2d(20,10,20,20,k_mat(7),1);
        [k8,ft8] = truss2d(20,20,20,30,k_mat(8),1);
        [k9,ft9] = truss2d(20,30,20,40,k_mat(9),1);
        [k10,ft10] = truss2d(20,40,20,50,k_mat(10),1);
        [k11,ft11] = truss2d(30,0,30,10,k_mat(11),1);
        [k12,ft12] = truss2d(30,10,30,20,k_mat(12),1);
        [k13,ft13] = truss2d(30,20,30,30,k_mat(13),1);
        [k14,ft14] = truss2d(30,30,30,40,k_mat(14),1);
        [k15,ft15] = truss2d(30,40,30,50,k_mat(15),1);
        [k16,ft16] = truss2d(40,0,40,10,k_mat(16),1);
        [k17,ft17] = truss2d(40,10,40,20,k_mat(17),1);
        [k18,ft18] = truss2d(40,20,40,30,k_mat(18),1);
        [k19,ft19] = truss2d(40,30,40,40,k_mat(19),1);
        [k20,ft20] = truss2d(40,40,40,50,k_mat(20),1);
        [k21,ft21] = truss2d(0,10,10,10,k_mat(21),1);
        [k22,ft22] = truss2d(10,10,20,10,k_mat(22),1);
        [k23,ft23] = truss2d(20,10,30,10,k_mat(23),1);
        [k24,ft24] = truss2d(30,10,40,10,k_mat(24),1);
        [k25,ft25] = truss2d(40,10,50,10,k_mat(25),1);
        [k26,ft26] = truss2d(0,20,10,20,k_mat(26),1);
        [k27,ft27] = truss2d(10,20,20,20,k_mat(27),1);
        [k28,ft28] = truss2d(20,20,30,20,k_mat(28),1);
        [k29,ft29] = truss2d(30,20,40,20,k_mat(29),1);
        [k30,ft30] = truss2d(40,20,50,20,k_mat(30),1);
        [k31,ft31] = truss2d(0,30,10,30,k_mat(31),1);
        [k32,ft32] = truss2d(10,30,20,30,k_mat(32),1);
        [k33,ft33] = truss2d(20,30,30,30,k_mat(33),1);
        [k34,ft34] = truss2d(30,30,40,30,k_mat(34),1);
        [k35,ft35] = truss2d(40,30,50,30,k_mat(35),1);
        [k36,ft36] = truss2d(0,40,10,40,k_mat(36),1);
        [k37,ft37] = truss2d(10,40,20,40,k_mat(37),1);
        [k38,ft38] = truss2d(20,40,30,40,k_mat(38),1);
        [k39,ft39] = truss2d(30,40,40,40,k_mat(39),1);
        [k40,ft40] = truss2d(40,40,50,40,k_mat(40),1);
               
        
        
                
        kg  = zeros(32,32);
        [kg] = addk(kg,k1,id,1);
        [kg] = addk(kg,k2,id,2);
        [kg] = addk(kg,k3,id,3);
        [kg] = addk(kg,k4,id,4);
        [kg] = addk(kg,k5,id,5);
        [kg] = addk(kg,k6,id,6);
        [kg] = addk(kg,k7,id,7);
        [kg] = addk(kg,k8,id,8);
        [kg] = addk(kg,k9,id,9);
        [kg] = addk(kg,k10,id,10);
        [kg] = addk(kg,k11,id,11);
        [kg] = addk(kg,k12,id,12);
        [kg] = addk(kg,k13,id,13);
        [kg] = addk(kg,k14,id,14);
        [kg] = addk(kg,k15,id,15);
        [kg] = addk(kg,k16,id,16);
        [kg] = addk(kg,k17,id,17);
        [kg] = addk(kg,k18,id,18);
        [kg] = addk(kg,k19,id,19);
        [kg] = addk(kg,k20,id,20);
        [kg] = addk(kg,k21,id,21);
        [kg] = addk(kg,k22,id,22);
        [kg] = addk(kg,k23,id,23);
        [kg] = addk(kg,k24,id,24);
        [kg] = addk(kg,k25,id,25);
        [kg] = addk(kg,k26,id,26);
        [kg] = addk(kg,k27,id,27);
        [kg] = addk(kg,k28,id,28);
        [kg] = addk(kg,k29,id,29);
        [kg] = addk(kg,k30,id,30);
        [kg] = addk(kg,k31,id,31);
        [kg] = addk(kg,k32,id,32);
        [kg] = addk(kg,k33,id,33);
        [kg] = addk(kg,k34,id,34);
        [kg] = addk(kg,k35,id,35);
        [kg] = addk(kg,k36,id,36);
        [kg] = addk(kg,k37,id,37);
        [kg] = addk(kg,k38,id,38);
        [kg] = addk(kg,k39,id,39);
        [kg] = addk(kg,k40,id,40);


        % Form the global load vector
        fg = zeros(32,1);
        
        
        fg(1:2:end)=F_h;
        fg(2:2:end)=F_v;
        
%         fg(13)=F_1h;
%         fg(14)=F_1v;
%         fg(15)=F_2h;
%         fg(16)=F_2v;
%         fg(23)=F_3h;
%         fg(24)=F_3v;
%         fg(25)=F_4h;
%         fg(26)=F_4v;
%         
        
        % Solve for nodal displacements
        vg = kg\fg;
        vgx = vg(1:2:end);
        vgy = vg(2:2:end);
        
    end
    
    displacement = sqrt(vgx.^2+vgy.^2); 
    
    disp_vec = [displacement(6), displacement(11)];
    
%     disp_vec = [displacement(7), displacement(11)];
%     disp_vec = [displacement(10),displacement(7),displacement(11)];  
  

end
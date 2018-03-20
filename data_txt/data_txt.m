clc
clear
source_dir_train='F:\zjc\Barefoot_metric_learning\Data\V1.4.0.7\';
source_dir_test='F:\zjc\Barefoot_metric_learning\Data\V1.4.0.7\';
train_name='V1.4.0.7_700_train.txt';
test_name='V1.4.0.7_700_test.txt';
file_train=fopen(train_name,'w');
file_test=fopen(test_name,'w');
folder_name=dir(source_dir_train);
folder_name2=dir(source_dir_test);
for i=3:900
    i
    file_name=dir([source_dir_train,folder_name(i).name]);
    if length(file_name)<=3
        continue
    end
    if length(file_name)>10
       folder_path=[source_dir_train,folder_name(i).name];
       fprintf(file_train,'%s\n',folder_path);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=3:900
    file_name=dir([source_dir_test,folder_name2(i).name]);
    if length(file_name)<=5
        continue
    end
    if length(file_name)<=10
       folder_path=[source_dir_test,folder_name2(i).name];
       fprintf(file_test,'%s\n',folder_path);
    end
end
fclose(file_train);        
fclose(file_test);  



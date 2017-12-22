clc
clear
source_dir_train='\\Desktop-78cpgr4\e\PROJECT\Foot_Height\data_Foot_Height\Face_database\CASIA\CASIA-maxpy-clean\';
source_dir_test='\\Desktop-78cpgr4\e\PROJECT\Foot_Height\data_Foot_Height\Face_database\CASIA\test\';
train_name='face_casia_train.txt';
test_name='face_casia_test.txt';
file_train=fopen(train_name,'w');
file_test=fopen(test_name,'w');
folder_name=dir(source_dir_train);
folder_name2=dir(source_dir_test);
for i=3:length(folder_name)
    i
    file_name=dir([source_dir_train,folder_name(i).name]);
    if length(file_name)<=3
        continue
    end
    if length(file_name)>4
       folder_path=[source_dir_train,folder_name(i).name];
       fprintf(file_train,'%s\n',folder_path);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=3:length(folder_name2)
    file_name=dir([source_dir_test,folder_name2(i).name]);
    if length(file_name)<=3
        continue
    end
    if length(file_name)>4
       folder_path=[source_dir_test,folder_name2(i).name];
       fprintf(file_test,'%s\n',folder_path);
    end
end
fclose(file_train);        
fclose(file_test);  



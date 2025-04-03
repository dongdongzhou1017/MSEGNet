clc
clear all
close all

tic
%%
Path = '...\data\';
File = dir(fullfile(Path,'*.csv'));  % the format of data is .csv
FileNames = {File.name}';  
Length_Names = size(FileNames,1);  
fs = 128; % 

%% the construction fo P60s
for i = 1:Length_Names 
    tmp0 = [];
    name = strcat(Path, FileNames(i));
    temp = csvread(name{1,1});
    L = size(temp,1);
    for j = 2:L
        t0 = temp(j-1,:);  
        t1 = temp(j,:);
        t = [t0,t1];
        tmp0 = [tmp0;t];
    end
    tmp1 = [tmp1;tmp0];

    save_path = '...\Construction_data\P60s';
    file_name = strcat(save_path, FileNames(i));
    csvwrite(file_name{1,1}, tmp0)
end

%% the construction fo S60s
for i = 1:Length_Names 
    tmp0 = [];
    name = strcat(Path, FileNames(i));
    temp = csvread(name{1,1});
    L = size(temp,1);
    for j = 1:L-1
        t0 = temp(j,:);  
        t1 = temp(j+1,:);
        t = [t0,t1];
        tmp0 = [tmp0;t];
    end
    tmp1 = [tmp1;tmp0];

    save_path = '...\Construction_data\S60s';
    file_name = strcat(save_path, FileNames(i));
    csvwrite(file_name{1,1}, tmp0)
end

%% the construction fo C90s
for i = 1:Length_Names 
    tmp0 = [];
    name = strcat(Path, FileNames(i));
    temp = csvread(name{1,1});
    L = size(temp,1);
    for j = 2:L-1
        t0 = temp(j-1,:);  
        t1 = temp(j,:);
        t2 = temp(j+1,:);  
        t = [t0,t1,t2];
        tmp0 = [tmp0;t];
    end
    tmp1 = [tmp1;tmp0];

    save_path = '...\Construction_data\C90s';
    file_name = strcat(save_path, FileNames(i));
    csvwrite(file_name{1,1}, tmp0)
end
%% the construction fo C150s
for i = 1:Length_Names 
    tmp0 = [];
    name = strcat(Path, FileNames(i));
    temp = csvread(name{1,1});
    L = size(temp,1);
    for j = 1:L-4
        t0 = temp(j,:);  
        t1 = temp(j+1,:);
        t2 = temp(j+2,:);  
        t3 = temp(j+3,:);
        t4 = temp(j+4,:);
        t = [t0,t1,t2,t3,t4];
        tmp0 = [tmp0;t];
    end
    tmp1 = [tmp1;tmp0];

    save_path = '...\Construction_data\C150s';
    file_name = strcat(save_path, FileNames(i));
    csvwrite(file_name{1,1}, tmp0)
end
%%
toc

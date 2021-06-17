clear all;
clc;
load net

voicePath='D:\OneDrive - agu.edu.tr\Documents\MATLAB\Bionluk\voice_recognizer\esra\voice2.m4a';
voicePath2='D:\OneDrive - agu.edu.tr\Documents\MATLAB\Bionluk\voice_recognizer\news\news1.mp3';

voices = zeros(1,1872);

[y,Fs] = audioread(voicePath);
d=fdesign.highpass('Fst,Fp,Ast,Ap',8700/16000,10000/16000,40,1);% yuksek gecirgen filtre ortamdaki gurultuyu silmek icin uygulandı 
hp=design(d); 
x=filter(hp,y); 
[row,col] = size(x);
zeroVecLength = 20*Fs-row;
x=x';
x=[x,zeros(1,zeroVecLength)];
x=x';
aFE = audioFeatureExtractor("SampleRate",Fs, ...
"SpectralDescriptorInput","barkSpectrum", ...
"spectralCentroid",true, ...
"spectralKurtosis",true, ...
"pitch",true);
features = extract(aFE,x);
features(isnan(features))=0;

features = (features - mean(features,1))./std(features,[],1);
features = features(:);
voices(1,:)=features;
voices=voices';
  
X=sim(net,voices);

[max_num,max_idx] = max(X());

if max_idx <= 5
    disp("bu esra Saray'in sesi")
else
    disp("bu ornekler Esra Saray'a ait degil")
end

voices2 = zeros(1,1872);
[y2,Fs2] = audioread(voicePath2);
d=fdesign.highpass('Fst,Fp,Ast,Ap',8700/16000,10000/16000,40,1);% yuksek gecirgen filtre ortamdaki gurultuyu silmek icin uygulandı 
hp=design(d); 
x2=filter(hp,y2); 
[row,col] = size(x2);
zeroVecLength = 20*Fs;
x2=x2(1:zeroVecLength,1);
aFE = audioFeatureExtractor("SampleRate",Fs, ...
        "SpectralDescriptorInput","barkSpectrum", ...
        "spectralCentroid",true, ...
        "spectralKurtosis",true, ...
        "pitch",true);
features = extract(aFE,x2);
features(isnan(features))=0;
features = (features - mean(features,1))./std(features,[],1);
features = features(:);
voices2(1,:)=features';

voices2=voices2';
X2=sim(net,voices2);
%% if else block to detect voice
[max_num,max_idx2] = max(X2());
if max_idx2 <= 5
    disp("bu esra Saray'in sesi")
else
    disp("bu ornekler Esra Saray'a ait degil")
end



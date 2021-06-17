clc;
clear all;
close all;
dosyayeri='D:\OneDrive - agu.edu.tr\Documents\MATLAB\Bionluk\voice_recognizer\esra\';
dosyaturu='.m4a';

%burada klasörün içindeki jpeg uzantılı dosyaları alıyoruz 
icerik = dir([dosyayeri,'*',dosyaturu]);

sesSayisi = size(icerik,1);
voices = zeros(sesSayisi,1872);

counter=0;
for k=1:sesSayisi
    counter=counter+1;
    string = [dosyayeri,icerik(k,1).name];
    [y,Fs] = audioread(string);
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
    voices(k,:)=features;
end


dosyayeri='D:\OneDrive - agu.edu.tr\Documents\MATLAB\Bionluk\voice_recognizer\news\';
dosyaturu='.mp3';

%burada klasörün içindeki jpeg uzantılı dosyaları alıyoruz 
icerik = dir([dosyayeri,'*',dosyaturu]);

sesSayisi2 = size(icerik,1);
voices2 = zeros(sesSayisi2,1872);

counter=0;
for m=1:sesSayisi2
    counter=counter+1;
    string = [dosyayeri,icerik(k,1).name];
    [y,Fs2] = audioread(string);
    d=fdesign.highpass('Fst,Fp,Ast,Ap',8700/16000,10000/16000,40,1);% yuksek gecirgen filtre ortamdaki gurultuyu silmek icin uygulandı 
    hp=design(d); 
    x=filter(hp,y); 
    [row,col] = size(x);
    zeroVecLength = 20*Fs;
    x=x(1:zeroVecLength,1);
    aFE = audioFeatureExtractor("SampleRate",Fs, ...
        "SpectralDescriptorInput","barkSpectrum", ...
        "spectralCentroid",true, ...
        "spectralKurtosis",true, ...
        "pitch",true);
    features = extract(aFE,x);
    features(isnan(features))=0;

    features = (features - mean(features,1))./std(features,[],1);
    features = features(:);
    voices2(m,:)=features';
end

allVoices = [voices;voices2];
giris = allVoices';
giris(isnan(giris))=0;
target = eye(sesSayisi+sesSayisi2);%[1,0;1,0;1,0;1,0;1,0;0,1;0,1;0,1;0,1;0,1];%eye(sesSayisi+sesSayisi2);
[R,Q]=size(giris);
[S2,Q]=size(target);
S1=10;

%burada yapay sinir ağlarımızı logaritmik sigmoid şekilde egitiyoruz
%yukarıda aldığımız S1 ve S2 değerlerini burada yerine yazarsak yaklaşık
%300000 tane inputumuz ve 9 tane outputumuz olacak
net = newff(minmax(giris),[S1 S2],{'logsig' 'logsig'},'trainscg');
net.trainParam.per='sse';
net.trainParam.epochs=100000;
net.trainParam.goal=1e-5;
net.trainParam.min_grad = 1e-8;
net=train(net,giris,target);

%son olarak eğittiğimiz ağı kaydediyoruz
save net


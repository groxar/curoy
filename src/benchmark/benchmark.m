function results = multBench ()
for i=1:50
tic; rand(i*50) t=toc;
t
endfor
for i=1:50
tic; rand(i*50).+rand(i*50); t=toc;
t
endfor
for i=1:50
tic; rand(i*50)*rand(i*50); t=toc;
t
endfor

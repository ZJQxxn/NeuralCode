Task brief: 
For each trial, monkeys were required to fixate on the fixation point to launch the task. After 500 ms two shapes,trianlge and square, were presented on the both sides of fixation point. Monkeys could freely look at each shapes. After another 1000 ms, the fixation point disappear denoted as go signal, then monkeys were required to saccade to one shapes and matain fixation on their choice for 500 ms. Then came the reward feedback.
 In each r_state, one shape stably had higher reward probability. If the r_state change, the other shape had higher reward probability. 

File discription.

bhv_ana.mat
BHV: raw behavioural data

postion: triangle's position, left -1, right 1.

c_lr: monkeys' choice, left -1, right  1.

rwd_fed: reward feedback.whether monkey got reward in current trial. no reward -1, reward 1.

c_cue: monkeys' choice on which shape. triangle 1, square -1.

rprob : reward probability of high value shape.

r_state: 1 means triangle had higher reward probability. -1 means square had higher reward probability

****************************************************************************************************************************************


NeuroResponse.mat
The sampling frequency is 10 Hz.

Fd1(neuron*frame*trial): All trials were aligned to trial onset when the monkeys fixated on the fixation point.

Fd2(neuron*frame*trial): All trials were aligned to the end the choice, i.e. the starting point of reward feedback.





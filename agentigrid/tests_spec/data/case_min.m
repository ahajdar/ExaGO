function mpc = case_min
% CASE_MIN  Minimal synthetic case for QA tests (SPEC-PARSER).
mpc.version = '2';
mpc.baseMVA = 100;

%% bus data
mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	138	1	1.1	0.9;
	2	1	50	10	0	0	1	1.0	0	138	1	1.1	0.9;
];

%% generator data
mpc.gen = [
	1	40	0	30	-30	1.0	100	1	100	0;
];

%% branch data
mpc.branch = [
	1	2	0.01	0.1	0.02	250	250	250	0	0	1	-360	360;
];

%% generator cost data
mpc.gencost = [
	2	0	0	3	0.01	20	100;
];

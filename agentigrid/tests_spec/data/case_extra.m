function mpc = case_extra
% QA-HEADER-LINE synthetic case with extra sections
% second header line

mpc.version = '2';
mpc.baseMVA = 100;

% QA-COMMENT-BUS
mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	138	1	1.1	0.9;
	2	1	50	10	0	0	1	1.0	0	138	1	1.1	0.9;
	3	2	20	5	0	0	2	1.0	0	138	1	1.1	0.9;
];

% QA-COMMENT-GEN
mpc.gen = [
	1	40	0	30	-30	1.0	100	1	100	0;
	3	20	0	15	-15	1.02	100	1	60	10;
];

% QA-COMMENT-BRANCH
mpc.branch = [
	1	2	0.01	0.1	0.02	250	250	250	0	0	1	-360	360;
	2	3	0.02	0.2	0.04	150	150	150	0	0	1	-360	360;
];

% QA-COMMENT-GENCOST
mpc.gencost = [
	2	0	0	3	0.01	20	100;
	2	0	0	3	0.02	25	50;
];

% QA-COMMENT-GENFUEL
mpc.genfuel = {
	'coal';
	'ng';
};

mpc.gentype = {
	'ST';
	'GT';
};

mpc.bus_name = {
	'Bus One';
	'Bus Two';
	'Bus Three';
};

mpc.areas = [
	1	1;
	2	3;
];

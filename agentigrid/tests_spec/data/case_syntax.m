function mpc = case_syntax
% CASE_SYNTAX  Same data as case_plain.m, written with every accepted syntax variant.
mpc.version = '2';
mpc.baseMVA = 100;

mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	138	1	1.1	0.9;	% inline comment
	% whole-line comment inside a matrix

  2   1  50   10 	0  0  1  1  0  138  1  1.1  0.9
	3.0	2.0	-5	1.5e-3	0	0	1.0	1	0	1.38E2	1	1.1	0.9;
];

mpc.gen = [
	1	40	0	inf	-INF	1	100	1.0	100	0
	3	20	0	nan	-15	1	100	1	60	0;   % lower-case nan
];
mpc.branch = [
	1	2	1e-2	0.1	2E-2	250	250	250	0	0	1	-360	360;
	2.0	3.0	0.02	0.2	0.04	0	0	0	0	0	1.0	-360	360;
];
mpc.gencost = [
	2	0	0	3	0.01	20	100
	2.0	0	0	3.0	0.02	25	50;
];

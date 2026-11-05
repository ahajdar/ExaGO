function mpc = case_nogen
% CASE_NOGEN  Only a bus section: gen, branch and gencost are absent.
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	138	1	1.1	0.9;
];

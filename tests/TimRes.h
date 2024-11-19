/*
--------------------------------------------------------------------------------
-- Company:        IRFU / CEA Saclay
-- Engineers:      Irakli.MANDJAVIDZE@cea.fr (IM)
-- 
-- Project Name:   Clas12 Micromegas Vertex Tracker
-- Design Name:    Common accross designs
--
-- Module Name:    TimRes.h
-- Description:    Evaluation of timing resolution
--
-- Target Devices: Win / Linux PC
-- Tool versions:  Win VS / Linux GNU
-- 
-- Create Date:    0.0 2013/10/17 IM
-- Revision:        
--
-- Comments:
--
--------------------------------------------------------------------------------
*/

#ifndef H_TimRes
#define H_TimRes

	#include "bitmanip.h"
	#include "FeuData.h"
	#include "PedHisto.h"

	extern int max_value[D_Pd_Nb_Of_Histos];
	extern int max_sample[D_Pd_Nb_Of_Histos];
	extern int fsat[D_Pd_Nb_Of_Histos];
	extern int lsat[D_Pd_Nb_Of_Histos];
	extern int smp_period_ns;
	extern int sparse_readout;
	extern int trg_period_ns;
	extern int min_tot;
	extern int max_tot;

	// buffer to keep event samples for all channels
	extern unsigned short sample_table[D_Fd_Dreams_Per_Feu*D_Fd_Nb_Dream_Chan][D_Fd_Max_Nb_Dream_Samples];

	int TimRes_Init();
	int TimRes_Calc(int ftstp, int max_nb_of_samples);
	int TimRes_ClusterUpdate();
#endif // #ifndef H_TimRes

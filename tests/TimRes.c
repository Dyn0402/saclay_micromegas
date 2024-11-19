/*
--------------------------------------------------------------------------------
-- Company:        IRFU / CEA Saclay
-- Engineers:      Irakli.MANDJAVIDZE@cea.fr (IM)
-- 
-- Project Name:   Clas12 Micromegas Vertex Tracker
-- Design Name:    Common accross designs
--
-- Module Name:    TimRes.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "TimRes.h"
#include "FeuData.h"
#include "PedHisto.h"

// buffer to keep event samples for all channels
unsigned short sample_table[D_Fd_Dreams_Per_Feu*D_Fd_Nb_Dream_Chan][D_Fd_Max_Nb_Dream_Samples];

// Reference channel
int    ref_chan_id;
double ref_chan_max_tim;

extern int    event_cnt;

extern int first_sample;
extern int last_sample;

unsigned int event_adc_count;
unsigned int event_mlt_count;

extern int max_fit[D_Pd_Nb_Of_Histos];

void CalcParabolaVertex(int x1, int y1, int x2, int y2, int x3, int y3, double *xv, double *yv)
{
	double denom;
	double A;
	double B;
	double C1;
	double C2;
	double C3;
	double C4;
	double C;

	denom = ((double)x1 - (double)x2) * ((double)x1 - (double)x3) * ((double)x2 - (double)x3);
	A = ((double)x3 * ((double)y2 - (double)y1) + (double)x2 * ((double)y1 - (double)y3) + (double)x1 * ((double)y3 - (double)y2)) / denom;
	B = ((double)x3*(double)x3 * ((double)y1 - (double)y2) + (double)x2*(double)x2 * ((double)y3 - (double)y1) + (double)x1*(double)x1 * ((double)y2 - (double)y3)) / denom;
	C1 = (double)x2 * (double)x3 * ((double)x2 - (double)x3) * (double)y1 ;
	C2 = (double)x3 * (double)x1 * ((double)x3 - (double)x1) * (double)y2 ;
	C3 = (double)x1 * (double)x2 * ((double)x1 - (double)x2) * (double)y3 ;
	C4 = C1 + C2 + C3;
	C = C4 / denom;
	*xv = (-B / (2*A));
	*yv = (C - B*B / (4*A));
	if( (*yv <= 0) )
		fprintf( stderr, "%s: denom=%f a=%f b=%f c=%f c1=%f c2=%f c3=%f c4=%f\n\r", __FUNCTION__, denom, A, B, C, C1, C2, C3, C4 );
}


// Initialize FeuData structure
int TimRes_Init()
{
	int feu_chan;
	int sample;
	for( feu_chan=0; feu_chan<D_Fd_Dreams_Per_Feu*D_Fd_Nb_Dream_Chan; feu_chan++ )
	{
		for( sample=0; sample<D_Fd_Max_Nb_Dream_Samples; sample++ )
			sample_table[feu_chan][sample] = -1;
	}
	ref_chan_id = -1;
	event_adc_count = 0;
	event_mlt_count = 0;
	return 0;
}

int TimRes_Calc(int ftstp, int max_nb_of_samples)
{
	int feu_chan;
	int max_sample_id;

	int sample1_tim;
	int sample1_val;
	int sample2_tim;
	int sample2_val;
	int sample3_tim;
	int sample3_val;
	double max_tim;
	double max_val;
	double diff_tim;
//fprintf(stderr, "%s: ftstp=%d max_nb_of_samples=%d\n", __FUNCTION__, ftstp, max_nb_of_samples);
	for( feu_chan=0; feu_chan<D_Fd_Dreams_Per_Feu*D_Fd_Nb_Dream_Chan; feu_chan++ )
	{
		if
		(
			( lsat[feu_chan] - fsat[feu_chan] >= min_tot )
			&&
			( lsat[feu_chan] - fsat[feu_chan] <= max_tot )
			&&
			( fsat[feu_chan] >= first_sample )
			&&
			( lsat[feu_chan] <= last_sample )
		)
		{
			max_sample_id = max_sample[feu_chan];
			if( (0<max_sample_id) && (max_sample_id<max_nb_of_samples-1) )
			{
				if( max_value[feu_chan] != sample_table[feu_chan][max_sample_id] )
				{
					fprintf(stderr, "TimRes_Calc: sample_table[%d][%d]=%d != max_value[%d]=%d next=%d prev=%d\n",
						feu_chan, max_sample_id, sample_table[feu_chan][max_sample_id],
						feu_chan, max_value[feu_chan],
						sample_table[feu_chan][max_sample_id+1], sample_table[feu_chan][max_sample_id-1]);
					return -1;
				}
				sample1_tim = (max_sample_id-1) * smp_period_ns * (sparse_readout+1);
				sample1_val = sample_table[feu_chan][max_sample_id-1];
				sample2_tim = max_sample_id * smp_period_ns * (sparse_readout+1);
				sample2_val = sample_table[feu_chan][max_sample_id];
				sample3_tim = (max_sample_id+1) * smp_period_ns * (sparse_readout+1);
				sample3_val = sample_table[feu_chan][max_sample_id+1];
				CalcParabolaVertex
				(
					sample1_tim, sample1_val,
					sample2_tim, sample2_val,
					sample3_tim, sample3_val,
					&max_tim,    &max_val
				);
				if( max_val > 4095. )
					max_val = 4095.;

				if( (max_tim <= 0) || (max_val <= 0) )
				{
					fprintf(stderr, "%s: feu_chan=%d ftstp=%d max_tim=%d norm_max_tim=%d max_val=%d\n",
						__FUNCTION__, feu_chan, ftstp, (int)(max_tim+0.5), (int)(max_tim+ftstp+0.5), (int)(max_val+0.5));
					fprintf
					(
						stderr,
						"%s: tim1=%d val1=%d tim2=%d val2=%d tim3=%d val3=%d\n",
						__FUNCTION__,
						sample1_tim, sample1_val,
						sample2_tim, sample2_val,
						sample3_tim, sample3_val
					);
				}
				else
				{
					TimRes_Histo_Add( feu_chan, ftstp, (int)(max_tim        +0.5), (int)(max_val+0.5) );
//					if( ftstp == 0 )
//						TimRes_Histo_Add( feu_chan, 7, (int)(max_tim+8*trg_period_ns+0.5), (int)(max_val+0.5) );
//					else
					TimRes_Histo_Add( feu_chan, 7, (int)(max_tim+ftstp*trg_period_ns+0.5), (int)(max_val+0.5) );
					max_fit[feu_chan] = (int)(max_val + 0.5);
					if(max_fit[feu_chan]<0)
						fprintf(stderr, "%s: sample_table[%d][%d]=%d != max_value[%d]=%d next=%d prev=%d max_fit=%d max_val=%f\n",
							__FUNCTION__,
							feu_chan, max_sample_id, sample_table[feu_chan][max_sample_id],
							feu_chan, max_value[feu_chan],
							sample_table[feu_chan][max_sample_id + 1], sample_table[feu_chan][max_sample_id - 1],
							max_fit[feu_chan], max_val);
					event_adc_count += ((int)(max_val+0.5));
					event_mlt_count++;
					if( ref_chan_id == -1 )
					{
						ref_chan_id = feu_chan;
						ref_chan_max_tim = max_tim;
					}
					else
					{
						diff_tim = 100. + max_tim - ref_chan_max_tim;
						if( diff_tim < 0 )
							diff_tim = -diff_tim;
						TimRes_Histo_AddDiff( ftstp, (int)(diff_tim+0.5) );
						TimRes_Histo_AddDiff( 7,     (int)(diff_tim+0.5) );
					}
				} // else of if( (max_tim < 0) || (max_val < 0) || (4095<max_val) )
/*
			else
			{
				if( feu_chan == 130 )
fprintf(stderr, "%s: feu_chan=%d max_sample_id=%d out of range\n", __FUNCTION__, feu_chan, max_sample_id);
			} // if( (0<max_sample_id) && (max_sample_id<max_nb_of_samples-1) )
*/
			} // if( (0<max_sample_id) && (max_sample_id<max_nb_of_samples-1) )
/*
		else
		{
			if( feu_chan == 130 )
fprintf(stderr, "%s: feu_chan=%d lsat=%d fsat=%d diff < min_tot=%d for event=%d\n", __FUNCTION__, feu_chan, lsat[feu_chan], fsat[feu_chan], min_tot, event_cnt);
*/
		} // if( lsat[feu_chan] - fsat[feu_chan] >= min_tot )
	} // for( feu_chan=0; feu_chan<D_Fd_Dreams_Per_Feu*D_Fd_Nb_Dream_Chan; feu_chan++ )
	return(0);
}

int TimRes_ClusterUpdate()
{
	if( event_adc_count )
	{
		TimRes_Histo_ClusterUpdate( event_adc_count, event_mlt_count );
	}
	return(0);
}

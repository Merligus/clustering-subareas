#include "Python.h"
// compile with
// gcc -pthread -fno-strict-aliasing -fwrapv -Wall -Wstrict-prototypes -fPIC -std=c99 -O3 -fomit-frame-pointer -Isrc/ -I/usr/include/python3.8 -shared -o c_functions.so c_functions.c


void weightedMean(double *y, double *sumwvec, double *target, double *w, long *iord, long* ties, long n, long nties)
{
	long k, l, ind, nprevties;
	double sumw, sumwt;

	nprevties = 0;
	for (k = 0; k < nties; k++)
	{
		sumwt = 0;
		sumw  = 0;   
		for (l = 0; l < ties[k]; l++)
		{
			ind = iord[nprevties + l];
			sumwt += w[ind]*target[ind];
			sumw += w[ind];
		}
		if (sumw > 1E-10)
			y[k] = sumwt / sumw;
		else
			y[k] = 0;

		sumwvec[k] = sumw;
		nprevties += ties[k];
	}
}

static PyObject *wmonreg(double *disp, double *w, long n)
/* Weighted monotone regression 
 disp is a target vector of length n that should be ordered monotonically increasing
 The up-and-down blocks algorithm below transforms disp such that it is weighted least-squares
 optimal to the target vector 
 
 Input:
 disp: vector of n with target vector
 w:    vector of n with nonnegative weights
 
 Output:
 disp: vector of n with results of weighted monotone regression. 
 
 Authors: Patrick Groenen and Gertjan van den Burg
 Date:    June 5, 2014
*/
{
	long j, iup, luph, idown, lovbkh;
	double sw, sds, wovbkh, trialv;

	lovbkh = 0;                             // Length of block
	wovbkh = 0;                             // Sum of weights of a block for which order restriction is active 
	luph = -1;                              // Index of start of block

	for (iup = 1; iup < n; iup++) // Loop over second to last element of disp
	{
		if (disp[iup] < disp[iup - 1]) {      // If previous element is larger
			sds   = w[iup] * disp[iup];         // Initialize weighted sum of disp
			sw    = w[iup];                     // Initialize sum of weights 
			idown = iup;
			do // Check for all ellements of lower than iup-1
			{
				idown--;
				if (luph == idown) // Use this if there is already a block downwards (speeds up computation)
				{
					sds   += wovbkh * disp[luph];   // Add weighted sum of block so far to weighted sum of disp
					sw    += wovbkh;                // Add sum of weights of block so far to sum of weights
					idown = idown - lovbkh + 0;     // Skip counter idown by the length of the current block
				}
				else
				{
					sds += w[idown] * disp[idown];  // Add element disp[idown] to weighted sum of disp
					sw  += w[idown];                // Add w[idown] to the sum of weights
				}
				trialv = sds / sw;                // Compute trial of block average				
				if (idown == 0)
					break;                          // Stop loop if first element is reached
			} while (disp[idown - 1] > trialv); // Stop if previous element of disp is smaller than block average

			wovbkh = 0;
			for (j=idown; j <=iup; j++)
			{     
				disp[j] = trialv;                 // Fill block with block average (trialv)
				wovbkh += w[j];                   // Compute sum of weights of the block
			}
			lovbkh = iup - idown;               // Length of block
			luph   = iup;                       // Index of start of the block 
		}
	}
}
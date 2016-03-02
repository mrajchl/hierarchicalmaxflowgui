/***************************************************************************/
/* Name:       cmf3DCut_ovl.h                           
   Date:          Nov 07, 2011                                                 

   Authors:       Jing Yuan (cn.yuanjing@googlemail.com) 
                  Martin Rajchl (mrajchl@imaging.robarts.ca)  

   Description:   Fast Max-Flow Implementation for multilayered min-cut
   Inputs 
   (C_t, para, alpha): C_t - the capacities of n flows
                       para 0,1 - rows, cols
		       para 2 - n: the label number
		       para 3 - the total number of iterations
		       para 4 - the error criterion
		       para 5 - cc for ALM
		       para 6 - steps for each iteration
		       alpha - the vector of penalty parameters with n-1 elements
 
   Outputs
   (u, erriter, num): u - the computed labeling result 
                      erriter - the error at each iteration
		      num - the iteration on stop
 
   For more details, see the report:
   Egil Bae, Jing Yuan, Xue-Cheng Tai and Yuri Boykov
   "A FAST CONTINUOUS MAX-FLOW APPROACH TO NON-CONVEX MULTILABELING PROBLEMS" 
    Submitted to Math. Comp. (AMS Journals)

*/
/***************************************************************************/

#ifndef _CMF3DCUT_OVL_H_
#define _CMF3DCUT_OVL_H_


extern "C" void cmf3DCut_ovl(float *fCt, float *pars, float *lambda, float *pt, float *u);

#endif // _cmf3DCUT_OVL_H_

/***************************************************************************/
/* Name:      cmf3DCut_std                              

   Authors:
   Martin Rajchl   mrajchl@imaging.robarts.ca
   Jing Yuan       cn.yuanjing@googlemail.com

   Description:
   Parallelized Continuous MaxFlow GraphCuts using CUDA

   For more details, see the report:
   Jing Yuan et al.
   "A Study on Continuous Max-Flow and Min-Cut Approaches"
   CVPR, 2010

   Jing Yuan, et al.
   "A Study on Continuous Max-Flow and Min-Cut Approaches: Part I: Binary Labeling"
   UCLA CAM, 2010

   Date: 2011/09/29

*/
/***************************************************************************/

#ifndef _CMF3DCUT_STD_H_
#define _CMF3DCUT_STD_H_


extern "C" void cmf3DCut_std(float *lambda, float *fCs, float *fCt, float *pars);

#endif // _cmf3DCUT_STD_H_

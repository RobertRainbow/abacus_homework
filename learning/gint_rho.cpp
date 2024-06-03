#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "gint_k.h"
#include "module_basis/module_ao/ORB_read.h"
#include "grid_technique.h"
#include "module_base/ylm.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/blas_connector.h"
#include "module_base/timer.h"
#include "gint_tools.h"

void Gint::gint_kernel_rho(
	const int na_grid,      // Number of grids
	const int grid_index,   // Index of the current grid
	const double delta_r,   // Delta radius for calculation
	int* vindex,		// Index array for grid points
	const int LD_pool,	// Leading dimension pool
	Gint_inout *inout)	// Input/Output structure
{
	//prepare block information
	int * block_iw, * block_index, * block_size;
	bool** cal_flag;
	//retrieve block information based on grid and other parameters
	Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw, block_index, block_size, cal_flag);

	//evaluate psi on grids
	Gint_Tools::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
	Gint_Tools::cal_psir_ylm(*this->gridt, 
		this->bxyz, na_grid, grid_index, delta_r,
		block_index, block_size, 
		cal_flag,
		psir_ylm.ptr_2D);
	
	//loop over spin states
	for(int is=0; is<GlobalV::NSPIN; ++is)
	{
		//Initialize and zero out the array for psi density matrix(DM)
		Gint_Tools::Array_Pool<double> psir_DM(this->bxyz, LD_pool);
		ModuleBase::GlobalFunc::ZEROS(psir_DM.ptr_1D, this->bxyz*LD_pool);
		//Different calculations based on GAMMA_ONLY flag and CACULATION type
		if(GlobalV::GAMMA_ONLY_LOCAL)
		{
			if (GlobalV::CALCULATION == "get_pchg")
			{
				// Multiply psi by DM for the "get_pchg" calculation type
				Gint_Tools::mult_psi_DM(
					*this->gridt, this->bxyz, na_grid, LD_pool,
					block_iw, block_size,
					block_index, cal_flag,
					psir_ylm.ptr_2D,
					psir_DM.ptr_2D,
                    inout->DM[is], inout->if_symm);
			}
			else
			{
				// Multiply psi by DM using a different method
				Gint_Tools::mult_psi_DM_new(
					*this->gridt, this->bxyz, grid_index, na_grid, LD_pool,
					block_iw, block_size,
					block_index, cal_flag,
					psir_ylm.ptr_2D,
					psir_DM.ptr_2D,
                    this->DMRGint[is], inout->if_symm);
			}	
			
		}
		else
		{
			//calculating g_mu(r) = sum_nu rho_mu,nu psi_nu(r)
			Gint_Tools::mult_psi_DMR(
				*this->gridt, this->bxyz, grid_index, na_grid,
				block_index, block_size,
				cal_flag, 
				psir_ylm.ptr_2D,
				psir_DM.ptr_2D,
				inout->DM_R[is],
				this->DMRGint[is],
                inout->if_symm);
		}

		//do sum_mu g_mu(r)psi_mu(r) to get electron density on grid
		this->cal_meshball_rho(
			na_grid, block_index,
			vindex, psir_ylm.ptr_2D,
			psir_DM.ptr_2D, inout->rho[is]);
	}
	delete[] block_iw;
	delete[] block_index;
	delete[] block_size;
	for(int ib=0; ib<this->bxyz; ++ib)
	{
		delete[] cal_flag[ib];
	}
	delete[] cal_flag;
}

void Gint::cal_meshball_rho(
	const int na_grid, // Number of grids
	int* block_index,  // Block index array
	int* vindex,	   // Index array for grid points
	double** psir_ylm, // 2D array for psi values
	double** psir_DMR, // 2D array for psi density matrix values
	double* rho)	   // Array to store the resulting electron density
{		
	const int inc = 1;
	// sum over mu to get density on grid
	for(int ib=0; ib<this->bxyz; ++ib)
	{
		// Perform dot product to accumulate density values
		double r=ddot_(&block_index[na_grid], psir_ylm[ib], &inc, psir_DMR[ib], &inc);
		const int grid = vindex[ib];
		rho[ grid ] += r;
	}
}



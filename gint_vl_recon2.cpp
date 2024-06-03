#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "gint_k.h"
#include "module_basis/module_ao/ORB_read.h"
#include "grid_technique.h"
#include "module_base/ylm.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/blas_connector.h"
#include "module_base/timer.h"
//#include <mkl_cblas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif

void Gint::gint_kernel_vlocal(
	const int na_grid,
	const int grid_index,
	const double delta_r,
	double* vldr3,
	const int LD_pool,
	double* pvpR_in,
	hamilt::HContainer<double>* hR)
{
	//prepare block information
	int * block_iw, * block_index, * block_size;
	bool** cal_flag;
	Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw, block_index, block_size, cal_flag);
	
	//evaluate psi and dpsi on grids
	Gint_Tools::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
	Gint_Tools::cal_psir_ylm(*this->gridt, 
		this->bxyz, na_grid, grid_index, delta_r,
		block_index, block_size, 
		cal_flag,
		psir_ylm.ptr_2D);
	
	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
	const Gint_Tools::Array_Pool<double> psir_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index, cal_flag, vldr3, psir_ylm.ptr_2D);

	//integrate (psi_mu*v(r)*dv) * psi_nu on grid
	//and accumulates to the corresponding element in Hamiltonian
    if(GlobalV::GAMMA_ONLY_LOCAL)
    {
		if(hR == nullptr) hR = this->hRGint;
		this->cal_meshball_vlocal_gamma(
			na_grid, LD_pool, block_iw, block_size, block_index, grid_index, cal_flag,
			psir_ylm.ptr_2D, psir_vlbr3.ptr_2D, hR);
    }
    else
    {
        this->cal_meshball_vlocal_k(
            na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
            psir_ylm.ptr_2D, psir_vlbr3.ptr_2D, pvpR_in);
    }

    //release memories
	delete[] block_iw;
	delete[] block_index;
	delete[] block_size;
	for(int ib=0; ib<this->bxyz; ++ib)
	{
		delete[] cal_flag[ib];
	}
	delete[] cal_flag;

	return;
}

void Gint::gint_kernel_dvlocal(
	const int na_grid,
	const int grid_index,
	const double delta_r,
	double* vldr3,
	const int LD_pool,
	double* pvdpRx,
	double* pvdpRy,
	double* pvdpRz)
{
	//prepare block information
	int * block_iw, * block_index, * block_size;
	bool** cal_flag;
	Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw, block_index, block_size, cal_flag);
	
	//evaluate psi and dpsi on grids
	Gint_Tools::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
	Gint_Tools::Array_Pool<double> dpsir_ylm_x(this->bxyz, LD_pool);
	Gint_Tools::Array_Pool<double> dpsir_ylm_y(this->bxyz, LD_pool);
	Gint_Tools::Array_Pool<double> dpsir_ylm_z(this->bxyz, LD_pool);

	Gint_Tools::cal_dpsir_ylm(*this->gridt, this->bxyz, na_grid, grid_index, delta_r,	block_index, block_size, cal_flag,
		psir_ylm.ptr_2D, dpsir_ylm_x.ptr_2D, dpsir_ylm_y.ptr_2D, dpsir_ylm_z.ptr_2D);

	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
	const Gint_Tools::Array_Pool<double> psir_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index, cal_flag, vldr3, psir_ylm.ptr_2D);

	//integrate (psi_mu*v(r)*dv) * psi_nu on grid
	//and accumulates to the corresponding element in Hamiltonian
	this->cal_meshball_vlocal_k(
		na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
		psir_vlbr3.ptr_2D, dpsir_ylm_x.ptr_2D, pvdpRx);
	this->cal_meshball_vlocal_k(
		na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
		psir_vlbr3.ptr_2D, dpsir_ylm_y.ptr_2D, pvdpRy);
	this->cal_meshball_vlocal_k(
		na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
		psir_vlbr3.ptr_2D, dpsir_ylm_z.ptr_2D, pvdpRz);

    //release memories
	delete[] block_iw;
	delete[] block_index;
	delete[] block_size;
	for(int ib=0; ib<this->bxyz; ++ib)
	{
		delete[] cal_flag[ib];
	}
	delete[] cal_flag;

	return;
}

void Gint::gint_kernel_vlocal_meta(
	const int na_grid,
	const int grid_index,
	const double delta_r,
	double* vldr3,
	double* vkdr3,
	const int LD_pool,
	double* pvpR_in,
	hamilt::HContainer<double>* hR)
{
	//prepare block information
	int * block_iw, * block_index, * block_size;
	bool** cal_flag;
	Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw, block_index, block_size, cal_flag);

    //evaluate psi and dpsi on grids
	Gint_Tools::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
	Gint_Tools::Array_Pool<double> dpsir_ylm_x(this->bxyz, LD_pool);
	Gint_Tools::Array_Pool<double> dpsir_ylm_y(this->bxyz, LD_pool);
	Gint_Tools::Array_Pool<double> dpsir_ylm_z(this->bxyz, LD_pool);

	Gint_Tools::cal_dpsir_ylm(*this->gridt,
		this->bxyz, na_grid, grid_index, delta_r,
		block_index, block_size, 
		cal_flag,
		psir_ylm.ptr_2D,
		dpsir_ylm_x.ptr_2D,
		dpsir_ylm_y.ptr_2D,
		dpsir_ylm_z.ptr_2D
	);
	
	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
	const Gint_Tools::Array_Pool<double> psir_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index, cal_flag, vldr3, psir_ylm.ptr_2D);

	//calculating df_mu(r) = vofk(r) * dpsi_mu(r) * dv
	const Gint_Tools::Array_Pool<double> dpsix_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index, cal_flag, vkdr3, dpsir_ylm_x.ptr_2D);
	const Gint_Tools::Array_Pool<double> dpsiy_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index, cal_flag, vkdr3, dpsir_ylm_y.ptr_2D);	
	const Gint_Tools::Array_Pool<double> dpsiz_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index, cal_flag, vkdr3, dpsir_ylm_z.ptr_2D);

    if(GlobalV::GAMMA_ONLY_LOCAL)
    {
		if(hR == nullptr) hR = this->hRGint;
		//integrate (psi_mu*v(r)*dv) * psi_nu on grid
		//and accumulates to the corresponding element in Hamiltonian
		this->cal_meshball_vlocal_gamma(
			na_grid, LD_pool, block_iw, block_size, block_index, grid_index, cal_flag,
			psir_ylm.ptr_2D, psir_vlbr3.ptr_2D, hR);
		//integrate (d/dx_i psi_mu*vk(r)*dv) * (d/dx_i psi_nu) on grid (x_i=x,y,z)
		//and accumulates to the corresponding element in Hamiltonian
		this->cal_meshball_vlocal_gamma(
			na_grid, LD_pool, block_iw, block_size, block_index, grid_index, cal_flag,
			dpsir_ylm_x.ptr_2D, dpsix_vlbr3.ptr_2D, hR);
		this->cal_meshball_vlocal_gamma(
			na_grid, LD_pool, block_iw, block_size, block_index, grid_index, cal_flag,
			dpsir_ylm_y.ptr_2D, dpsiy_vlbr3.ptr_2D, hR);
		this->cal_meshball_vlocal_gamma(
			na_grid, LD_pool, block_iw, block_size, block_index, grid_index, cal_flag,
			dpsir_ylm_z.ptr_2D, dpsiz_vlbr3.ptr_2D, hR);
    }
    else
    {
        this->cal_meshball_vlocal_k(
            na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
            psir_ylm.ptr_2D, psir_vlbr3.ptr_2D, pvpR_in);
		this->cal_meshball_vlocal_k(
            na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
			dpsir_ylm_x.ptr_2D, dpsix_vlbr3.ptr_2D, pvpR_in);
		this->cal_meshball_vlocal_k(
            na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
			dpsir_ylm_y.ptr_2D, dpsiy_vlbr3.ptr_2D, pvpR_in);
		this->cal_meshball_vlocal_k(
            na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
			dpsir_ylm_z.ptr_2D, dpsiz_vlbr3.ptr_2D, pvpR_in);
    }

    //release memories
	delete[] block_iw;
	delete[] block_index;
	delete[] block_size;
	for(int ib=0; ib<this->bxyz; ++ib)
	{
		delete[] cal_flag[ib];
	}
	delete[] cal_flag;

	return;
}
		
bool cal_grid_range(
	int ia1,
	int ia2,
	int &first_ib,
	int &last_ib,
	int &ib_length,
	int &cal_pair_num,
	const int bxyz,
	const int na_grid,  				// how many atoms on this (i,j,k) grid
	const int LD_pool,				// dimension of the data pool
	const int*const block_iw,			// block_iw[na_grid], index of wave functions for each block
	const int*const block_size, 			// block_size[na_grid],	number of columns of a band
	const int*const block_index,		    	// block_index[na_grid+1], count total number of atomis orbitals
	const int grid_index,                       	// index of grid group, for tracing global atom index
	const bool*const*const cal_flag,	    	// cal_flag[this->bxyz][na_grid], whether the atom-grid distance is larger than cutoff
	const double*const*const psir_ylm,		// psir_ylm[this->bxyz][LD_pool], spherical harmonic value array
	const double*const*const psir_vlbr3,	    	// psir_vlbr3[this->bxyz][LD_pool], spherical harmonics multiplied by local potential energy arrays
	hamilt::HContainer<double>* hR)	    		// this->hRGint is the container of <phi_0 | V | phi_R> matrix element
{
	first_ib=0;					
	// find first grid satisfying the distance condition of the atom pair
        for(int ib=0; ib<bxyz; ++ib)
        {
        	if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
                {
                	first_ib=ib;
                        break;
                }
        }
        last_ib=0;
	// find last grid satisfying the distance condition of the atom pair
	for(int ib=bxyz-1; ib>=0; --ib)
        {
        	if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
                {
                	last_ib=ib+1;
                        break;
                }
        }
       	ib_length = last_ib-first_ib;
        if(ib_length<=0) return 0;	// if no grid satisfies
	// count the number of grids satisfying the pair's condition
        cal_pair_num=0;
        for(int ib=first_ib; ib<last_ib; ++ib)
        {
        	cal_pair_num += cal_flag[ib][ia1] && cal_flag[ib][ia2];
        }
	return 1;
}


void Gint::cal_meshball_vlocal_gamma(
	const int na_grid,  				// how many atoms on this (i,j,k) grid
	const int LD_pool,				// dimension of the data pool
	const int*const block_iw,			// block_iw[na_grid], index of wave functions for each block
	const int*const block_size, 			// block_size[na_grid],	number of columns of a band
	const int*const block_index,		    	// block_index[na_grid+1], count total number of atomis orbitals
	const int grid_index,                       	// index of grid group, for tracing global atom index
	const bool*const*const cal_flag,	    	// cal_flag[this->bxyz][na_grid], whether the atom-grid distance is larger than cutoff
	const double*const*const psir_ylm,		// psir_ylm[this->bxyz][LD_pool], spherical harmonic value array
	const double*const*const psir_vlbr3,	    	// psir_vlbr3[this->bxyz][LD_pool], spherical harmonics multiplied by local potential energy arrays
	hamilt::HContainer<double>* hR)	    		// this->hRGint is the container of <phi_0 | V | phi_R> matrix element
{
	// timer tick
	ModuleBase::TITLE("Gint_interface","cal_meshball_vlocal_gamma");
	ModuleBase::timer::tick("Gint_interface","cal_meshball_vlocal_gamma");
	const char transa='N', transb='T'; 	// transpose flag for matrix multiplication
	const double alpha=1, beta=1;		// coefficients of matrix multiplication
    	const int lgd_now = this->gridt->lgd;	// current local grid dimension, deprecated now

	const int mcell_index = this->gridt->bcell_start[grid_index];	// start position of current cell's adjacent atoms
	
	// main loop, calculate the contribution of each atom in current grid
	for(int ia1=0; ia1<na_grid; ++ia1)
	{
		const int iat1= this->gridt->which_atom[mcell_index + ia1];	// global position of the atom ia1
		const int iw1_lo=block_iw[ia1];					// start position of wave functions of atom ia1
		const int m=block_size[ia1];					// number of columns of atom ia1 band, for matrix multiplication
		// atom pair ( ia1, ia2 )
		for(int ia2=0; ia2<na_grid; ++ia2)
		{
			const int iat2= this->gridt->which_atom[mcell_index + ia2];
			const int iw2_lo=block_iw[ia2];
			// considering symmetry
			if(iw1_lo<=iw2_lo)
			{
                                int first_ib=0;
				int last_ib=0;
				int ib_length=0;
				int cal_pair_num=0;
				// calculate the BaseMatrix of <iat1, iat2, R> atom-pair
				hamilt::AtomPair<double>* tmp_ap = hR->find_pair(iat1, iat2);
#ifdef __DEBUG
				assert(tmp_ap!=nullptr);
#endif
				if(cal_grid_range(ia1, ia2, first_ib, last_ib, ib_length, cal_pair_num,
							this->bxyz, na_grid, LD_pool, block_iw, block_size,
							block_index, grid_index, cal_flag, psir_ylm,
							psir_vlbr3, hR)==0) continue;
				const int n=block_size[ia2];	// same as m
				//std::cout<<__FILE__<<__LINE__<<" "<<n<<" "<<m<<" "<<tmp_ap->get_row_size()<<" "<<tmp_ap->get_col_size()<<std::endl;
				
				// more than 1/4 grids need calculate
                		if(cal_pair_num>ib_length/4)
                		{
					// calculate <phi_0 | V | phi_R> in the grid range
                    			dgemm_(&transa, &transb, &n, &m, &ib_length, &alpha,
                        			&psir_vlbr3[first_ib][block_index[ia2]], &LD_pool,
                        			&psir_ylm[first_ib][block_index[ia1]], &LD_pool,
                        			&beta, tmp_ap->get_pointer(0), &n);
					//&GridVlocal[iw1_lo*lgd_now+iw2_lo], &lgd_now);   
                		}
                		else
                		{
                    			for(int ib=first_ib; ib<last_ib; ++ib)
                    			{
                        			if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
                        			{
                        				int k=1;                            
                            				dgemm_(&transa, &transb, &n, &m, &k, &alpha,
                                				&psir_vlbr3[ib][block_index[ia2]], &LD_pool,
                                				&psir_ylm[ib][block_index[ia1]], &LD_pool,
                                				&beta, tmp_ap->get_pointer(0), &n);                          
                        			}
                    			}
                		}
				//std::cout<<__FILE__<<__LINE__<<" "<<tmp_ap->get_pointer(0)[2]<<std::endl;
			}
		}
	}
	ModuleBase::timer::tick("Gint_interface","cal_meshball_vlocal_gamma");
}
int find_offset(const Gint* gint, const int id1, const int id2, const int iat1, const int iat2)
{
    const int R1x = gint->gridt->ucell_index2x[id1];
    const int R2x = gint->gridt->ucell_index2x[id2];
    const int dRx = R1x - R2x;
    const int R1y = gint->gridt->ucell_index2y[id1];
    const int R2y = gint->gridt->ucell_index2y[id2];
    const int dRy = R1y - R2y;
    const int R1z = gint->gridt->ucell_index2z[id1];
    const int R2z = gint->gridt->ucell_index2z[id2];
    const int dRz = R1z - R2z;

    const int index = gint->gridt->cal_RindexAtom(dRx, dRy, dRz, iat2);
    const int offset = gint->gridt->binary_search_find_R2_offset(index, iat1);

    assert(offset < gint->gridt->nad[iat1]);
    return offset;
}

void process_atom_pair(
    const Gint* gint,
    const int iat1,
    const int iat2,
    const int id1,
    const int id2,
    const int m,
    const int n,
    const int idx1,
    const int idx2,
    const int DM_start,
    const int* block_size,
    const int* block_index,
    bool** cal_flag,
    double** psir_ylm,
    double** psir_vlbr3,
    double* pvpR,
    int bxyz,
    int LD_pool)
{
    char transa = 'N', transb = 'T';
    double alpha = 1, beta = 1;

    int cal_num = 0;
    for (int ib = 0; ib < bxyz; ++ib)
    {
        if (cal_flag[ib][iat1] && cal_flag[ib][iat2])
            ++cal_num;
    }

    if (cal_num == 0) return;

    int offset = find_offset(gint, id1, id2, iat1, iat2);
    const int iatw = DM_start + gint->gridt->find_R2st[iat1][offset];

    if (cal_num > bxyz / 4)
    {
        int k = bxyz;
	#pragma omp critical
	{
        dgemm_(&transa, &transb, &n, &m, &k, &alpha,
               &psir_vlbr3[0][idx2], &LD_pool,
               &psir_ylm[0][idx1], &LD_pool,
               &beta, &pvpR[iatw], &n);
       	}
    }
    else
    {
        for (int ib = 0; ib < bxyz; ++ib)
        {
            if (cal_flag[ib][iat1] && cal_flag[ib][iat2])
            {
                int k = 1;
		#pragma omp critical
		{
                dgemm_(&transa,&transb,&n,&m,&k,&alpha,
		       &psir_vlbr3[ib][idx2], &LD_pool,
                       &psir_ylm[ib][idx1], &LD_pool,
                       &beta, &pvpR[iatw], &n);
		}
            }
        }
    }
}

void Gint::cal_meshball_vlocal_k(
    int na_grid,
    int LD_pool,
    int grid_index,
    int* block_size,
    int* block_index,
    int* block_iw,
    bool** cal_flag,
    double** psir_ylm,
    double** psir_vlbr3,
    double* pvpR)
{
    ModuleBase::TITLE("Gint_interface","cal_meshball_vlocal_k_recon");
    ModuleBase::timer::tick("Gint_interface","cal_meshball_vlocal_k_recon");

    int k = this->bxyz;
    
    #pragma omp parallel for schedule(dynamic)
    for (int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int idx1 = block_index[ia1];
        int m = block_size[ia1];
        const int mcell_index1 = this->gridt->bcell_start[grid_index] + ia1;
        const int iat1 = this->gridt->which_atom[mcell_index1];
        const int id1 = this->gridt->which_unitcell[mcell_index1];
        const int DM_start = this->gridt->nlocstartg[iat1];

        for (int ia2 = 0; ia2 < na_grid; ++ia2)
        {
            const int mcell_index2=this->gridt->bcell_start[grid_index]+ia2;
	    const int iat2 = this->gridt->which_atom[mcell_index2];
            if (iat1 <= iat2)
            {
                const int idx2 = block_index[ia2];
                int n = block_size[ia2];
                const int id2 = this->gridt->which_unitcell[mcell_index2];
                process_atom_pair(this, iat1, iat2, id1, id2, m, n, idx1, idx2, DM_start,
                                  block_size, block_index, cal_flag, psir_ylm, psir_vlbr3, pvpR,
                                  this->bxyz, LD_pool);
            }
        }
    }

    ModuleBase::timer::tick("Gint_interface","cal_meshball_vlocal_k_recon");
}

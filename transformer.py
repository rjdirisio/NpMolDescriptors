import numpy as np
import itertools as itt

from pyvibdmc.analysis import *


class Transformer:
    """Takes in Cartesian coordinates and efficiently outputs ML descriptors"""

    @staticmethod
    def f_c(nonzero_rijs, r_cut):
        """
        'cutoff function', a switching function that decays smoothely
        takes in (a stack of) rijs, and a stack of r_cuts
        """
        fc = 0.5 * (1 + np.cos(np.pi * nonzero_rijs / r_cut))
        fc[nonzero_rijs > r_cut] = 0.0
        return fc

    @staticmethod
    def get_g2(nonzero_distz, fcs, eta_2, r_s_2):
        expie = np.exp(-1 * eta_2 * (nonzero_distz - r_s_2) ** 2) #.reshape(-1, num_atoms, num_atoms - 1)
        expie -= np.eye(len(expie[0]))
        g_2 = np.sum(expie * fcs, axis=2)
        return g_2

    @staticmethod
    def get_g4(cds, num_atoms, nonzero_distz, fcs, zeta_4, eta_4,lam_4):
        analyzer = AnalyzeWfn(cds)
        atm_idxs = np.arange(num_atoms)
        # get all atom nums excluding current atom
        excluding_idx = np.array([atm_idxs[np.logical_not(atm_idxs == xnum)] for xnum in atm_idxs])
        theta_2 = len(list(itt.combinations(excluding_idx[0], 2)))  # number of bond angles we have for a given atom
        # initialize thetas, which is cds x num_atoms (all i's) and then number of angles for each i
        thetas = np.zeros((len(cds), num_atoms, theta_2))
        ex_sum = np.zeros(thetas.shape)
        fcs_ordered = np.zeros(thetas.shape)
        for atm_num in range(num_atoms):
            itter = itt.combinations(excluding_idx[atm_num], 2)
            for idx2, atm_grp in enumerate(itter):
                # print(atm_grp[0], atm_num, atm_grp[1])
                theta_jik = analyzer.bond_angle(atm_grp[0], atm_num, atm_grp[1])
                thetas[:, atm_num, idx2] = theta_jik
                sum_square = np.sum(np.square([nonzero_distz[:,atm_num,atm_grp[0]],
                              nonzero_distz[:,atm_num,atm_grp[1]],
                              nonzero_distz[:,atm_grp[0],atm_grp[1]]]),axis=0)
                ex_sum[:,atm_num, idx2] = sum_square
                fcs_ordered[:,atm_num, idx2] = fcs[:,atm_num,atm_grp[0]]*fcs[:,atm_num,atm_grp[1]]*fcs[:,atm_grp[0],atm_grp[1]]
        # Reshape nonzero_distz to be nx(n_atoms-1)xn_atoms to get all rijs for a given i
        expie = np.exp(-eta_4*ex_sum)
        total = 2**(1-zeta_4) * np.sum((1+lam_4*np.cos(thetas))**zeta_4 * expie * fcs_ordered,axis=-1)
        return total

    @staticmethod
    def acsf_it(cds, rcut, eta_2, r_s_2, zeta_4, eta_4, lam_4):
        num_atoms = cds.shape[1]
        rijs = Transformer.atm_atm_dists(cds, mat=True)
        rijs = rijs - np.eye(len(rijs[0]))
        fcs = Transformer.f_c(rijs, rcut)
        fcs -= np.eye(len(fcs[0]))
        # Get Gs
        g_1 = np.sum(fcs, axis=2)
        g_2 = Transformer.get_g2(rijs, fcs, eta_2, r_s_2)
        g_4 = Transformer.get_g4(cds, num_atoms, rijs, fcs, zeta_4, eta_4, lam_4)
        return g_1, g_2, g_4

    @staticmethod
    def atm_atm_dists(cds, mat=True):
        """
        Takes in coordinates and a boolean that specifies whether or not this will return a vector or matrix.
        This fills the diagonal with 1!!!
        """
        ngeoms = cds.shape[0]
        natoms = cds.shape[1]
        idxs = np.transpose(np.triu_indices(natoms, 1))
        atoms_0 = cds[:, tuple(x[0] for x in idxs)]
        atoms_1 = cds[:, tuple(x[1] for x in idxs)]
        diffs = atoms_1 - atoms_0
        dists = np.linalg.norm(diffs, axis=2)
        if mat:
            result = np.zeros((ngeoms, natoms, natoms))
            idxss = np.triu_indices_from(result[0], k=1)
            result[:, idxss[0], idxss[1]] = dists
            result[:, idxss[1], idxss[0]] = dists
            # fill diagonal with 1s
            result[:, np.arange(natoms), np.arange(natoms)] = 1
            return result
        else:
            return dists

    @staticmethod
    def sort_coulomb(c_mat):
        """Takes in coulomb matrix and sorts it according to the row norm"""
        indexlist = np.argsort(-1 * np.linalg.norm(c_mat, axis=1))
        sorted_c_mat = c_mat[np.arange(c_mat.shape[0])[:, None, None], indexlist[:, :, None], indexlist[:, None, :]]
        return sorted_c_mat

    @staticmethod
    def coulomb_it(cds, zs, sort_mat=True):
        """
        Takes in cartesian coordinates, outputs the upper triangle of
        the coulomb matrix.  Option to have it sorted for permutational invariance
        """
        # get 0.5 * z^0.4
        rest = np.ones((len(zs), len(zs)))
        np.fill_diagonal(rest, 0.5 * zs ** 0.4)
        # get zii^2/zij matrix
        zij = np.outer(zs, zs)
        # rij
        atm_atm_mat = Transformer.atm_atm_dists(cds, mat=True)
        coulomb = zij * rest / atm_atm_mat
        # sort each according to norm of rows/columns
        if sort_mat:
            coulomb_s = Transformer.sort_coulomb(coulomb)
        else:
            coulomb_s = coulomb
        idx = np.triu_indices_from(coulomb_s[1])
        upper_coulomb = coulomb_s[:, idx[0], idx[1]]
        return upper_coulomb


if __name__ == '__main__':
    from pyvibdmc.simulation_utilities import *
    import multiprocessing as mp
    nproc = 20
    nwalk = 1000000

    pool = mp.Pool(nproc)
    #bad ch5+ geoms
    walkers = np.random.random((nwalk,6,3))

    print(f"Nprocs: {nproc}")
    for i in range(10):
        start = time.time()
        lst = np.array_split(walkers,nproc)
        
        res = pool.starmap(Transformer.acsf_it,zip(lst,
                                          np.repeat(6.0,len(lst)),
                                          np.repeat(1.,len(lst)),
                                          np.repeat(0.,len(lst)),
                                          np.repeat(1.,len(lst)),
                                          np.repeat(1.,len(lst)),
                                          np.repeat(1.,len(lst))))
        water_acsf_rjd_mp = np.concatenate(res)
        print(f"Parallel ACSF Takes {time.time()-start}s")
    
    start = time.time()
    water_acsf_rjd = Transformer.acsf_it(walkers,
                                       rcut=6.0,
                                       eta_2=1,
                                       r_s_2=0,
                                       zeta_4=1,
                                       eta_4=1,
                                       lam_4=1)
    print(f"Single core ACSF Takes {time.time()-start}s")


    #coulomb timing
    for _ in range(10):
        start = time.time()
        zs = np.array([6,1,1,1,1,1])
        lst = np.array_split(walkers, nproc)
        res = pool.starmap(Transformer.coulomb_it, zip(lst, np.tile(zs,(len(lst),1))))
        water_acsf_rjd_mp = np.concatenate(res)
        print(f"Parallel Coulomb Takes {time.time() - start}s")

    start = time.time()
    water_acsf_rjd = Transformer.coulomb_it(walkers,zs)
    print(f"Single core Coulomb Takes {time.time() - start}s")

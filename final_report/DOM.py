'''
MAIN PROGRAM

    read user inputs
        LAI          - leaf area index
        f_dir        - fraction of direct solar radiation
        theta_solar  - solar zenith angle
        phi_solar    - solar azimuth angle
        rho_g        - ground reflectance
        rho_L        - leaf reflectance
        tau_L        - leaf transmittance
        N 
        M
        K
        TOL
        MAX_ITER
        F_in

    omega_L = rho_L + tau_L

    setup angular quadrature
        compute mu[1..N]
        compute w_mu[1..N]
        compute phi[1..M]
        compute w_phi

    compute extinction coefficient
        G = 0.5   for uniform leaf angle distribution

    compute layer thickness
        dL = LAI / K

    precompute scattering phase tables
        G_qq[n,m,i,j]
        G_sol[i,j]

    solve uncollided field
        get I0_dir[k]
        get I0[n,m,k]
        get fluxes from uncollided field
        get first collision source Q[i,j,k]

    solve collided field iteratively
        get IC[n,m,k]
        get collided fluxes

    combine uncollided + collided
        compute total fluxes
        compute scalar irradiance
        compute leaf absorption
        compute reflected flux
        compute transmitted flux
        compute energy balance

    optionally interpolate to arbitrary viewing directions
        compute top of canopy intensity / BRF / HDRF

    print summary tables and diagnostics
'''
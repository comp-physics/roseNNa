! 1-D acoustics through a bubbly region with a recurrent per-cell surrogate;
! the twin of bubbles.c. State arrays are hs/cs (Fortran is case-insensitive,
! and C collided with the cell index c); the slice copies are element loops,
! since a section assignment in a target region needs a runtime call the
! device lacks under amdflang.
program bubbles
    use bubbles_model, only: bubbles_infer
    use iso_fortran_env, only: real64
    use omp_lib, only: omp_get_wtime
    implicit none
#ifndef NB
#define NB 64
#endif
    integer, parameter :: nb_ = NB, nx = 512, ncell = nb_ * nx
    real(real64), parameter :: dx = 0.05_real64, dt = 0.05_real64
#ifndef NSTEPS
#define NSTEPS 400
#endif
    integer, parameter :: nsteps = NSTEPS
    real(real64), parameter :: beta0 = 0.1_real64                   ! unstable above ~0.2
    real(real64), parameter :: x_lo = 10.0_real64, x_hi = 20.0_real64, tol = 0.10_real64
    integer, parameter :: nbin = 8, n_sub = 10
    real(real64), parameter :: gam = 1.4_real64, mu = 0.05_real64
    integer, parameter :: hid = 32, nin = 1 + 2 * hid, nout = 1 + 2 * hid

    real(real64), allocatable :: wp(:), wm(:), wp2(:), wm2(:), pref(:), beta(:), src(:)
    real(real64), allocatable :: R(:,:), V(:,:), hs(:,:), cs(:,:)
    real(real64) :: t0, t_ref, t_nn, err
    integer :: b, i, c, k

    allocate(wp(ncell), wm(ncell), wp2(ncell), wm2(ncell), pref(ncell), beta(ncell), src(ncell))
    allocate(R(nbin, ncell), V(nbin, ncell), hs(hid, ncell), cs(hid, ncell))

    call set_pulse(wp, wm)
    do b = 1, nb_
        do i = 1, nx
            c = (b - 1) * nx + i
            beta(c) = merge(beta0, 0.0_real64, (i - 0.5_real64) * dx >= x_lo .and. (i - 0.5_real64) * dx <= x_hi)
            do k = 1, nbin
                R(k, c) = r0_of(k); V(k, c) = 0.0_real64
            end do
        end do
    end do
    hs = 0.0_real64; cs = 0.0_real64

    !$omp target enter data map(to: wp, wm, beta, R, V) map(alloc: wp2, wm2, src)
    t0 = omp_get_wtime()
    call run(wp, wm, wp2, wm2, beta, R, V, hs, cs, src, .false.)
    t_ref = omp_get_wtime() - t0
    !$omp target exit data map(from: wp, wm) map(delete: wp2, wm2, src, beta, R, V)
    pref = 0.5_real64 * (wp + wm)

    call set_pulse(wp, wm)
    !$omp target enter data map(to: wp, wm, beta, hs, cs) map(alloc: wp2, wm2, src)
    t0 = omp_get_wtime()
    call run(wp, wm, wp2, wm2, beta, R, V, hs, cs, src, .true.)
    t_nn = omp_get_wtime() - t0
    !$omp target exit data map(from: wp, wm) map(delete: wp2, wm2, src, beta, hs, cs)

    err = sqrt(sum((0.5_real64 * (wp + wm) - pref)**2) / sum(pref**2))
    print '(I0,A,I0,A,I0,A,I0,A,I0,A)', nb_, ' lines x ', nx, ' cells, ', nsteps, &
        ' steps; ', nbin, ' bins x ', n_sub, ' RK4 sub-steps per cell-step in the reference:'
    print '(A,F9.1,A,F6.1,A)', '  reference population  ', 1e3 * t_ref, ' ms  (', &
        1e9 * t_ref / (real(ncell, real64) * nsteps), ' ns per cell-step)'
    print '(A,F9.1,A,F6.1,A)', '  LSTM surrogate        ', 1e3 * t_nn, ' ms  (', &
        1e9 * t_nn / (real(ncell, real64) * nsteps), ' ns per cell-step)'
    print '(A,ES10.3)', '  relative L2 error of the surrogate''s pressure field: ', err
    if (.not. (err < tol)) then
        print '(A,F4.2)', 'FAIL: error above ', tol
        stop 1
    end if
    print '(A)', 'OK'

contains

    pure real(real64) function r0_of(k)     ! geomspace(0.5, 2, nbin), k = 1..nbin
        !$omp declare target
        integer, intent(in) :: k
        r0_of = 0.5_real64 * 4.0_real64**(real(k - 1, real64) / (nbin - 1))
    end function

    pure subroutine rp_rhs(r, v, p, r0, dr, dv)
        !$omp declare target
        real(real64), intent(in) :: r, v, p, r0
        real(real64), intent(out) :: dr, dv
        dr = v
        dv = ((r0 / r)**(3.0_real64 * gam) - 1.0_real64 - p - 4.0_real64 * mu * v / r - 1.5_real64 * v * v) / r
    end subroutine

    ! One acoustic step of one cell's bins; returns s = sum_k w_k 3 R_k^2 V_k / R0_k^3.
    function population_step(rr, vv, p) result(s)
        !$omp declare target
        real(real64), intent(inout) :: rr(nbin), vv(nbin)
        real(real64), intent(in) :: p
        real(real64) :: s, hh, r, v, r0, k1r, k1v, k2r, k2v, k3r, k3v, k4r, k4v
        integer :: k, sub
        hh = dt / n_sub
        s = 0.0_real64
        do k = 1, nbin
            r0 = r0_of(k); r = rr(k); v = vv(k)
            do sub = 1, n_sub
                call rp_rhs(r, v, p, r0, k1r, k1v)
                call rp_rhs(r + 0.5_real64 * hh * k1r, v + 0.5_real64 * hh * k1v, p, r0, k2r, k2v)
                call rp_rhs(r + 0.5_real64 * hh * k2r, v + 0.5_real64 * hh * k2v, p, r0, k3r, k3v)
                call rp_rhs(r + hh * k3r, v + hh * k3v, p, r0, k4r, k4v)
                r = r + hh / 6.0_real64 * (k1r + 2.0_real64 * k2r + 2.0_real64 * k3r + k4r)
                v = v + hh / 6.0_real64 * (k1v + 2.0_real64 * k2v + 2.0_real64 * k3v + k4v)
            end do
            rr(k) = r; vv(k) = v
            s = s + (1.0_real64 / nbin) * 3.0_real64 * r * r * v / r0**3
        end do
    end function

    subroutine step(wp, wm, wp_new, wm_new, beta, R, V, hs, cs, src, use_nn)
        real(real64), intent(in) :: wp(ncell), wm(ncell), beta(ncell)
        real(real64), intent(out) :: wp_new(ncell), wm_new(ncell), src(ncell)
        real(real64), intent(inout) :: R(nbin, ncell), V(nbin, ncell), hs(hid, ncell), cs(hid, ncell)
        logical, intent(in) :: use_nn
        real(real64) :: p, x(nin), y(nout)
        integer :: c, b, i, l, r_, j
        !$omp target teams distribute parallel do private(p, x, y, j)
        do c = 1, ncell
            p = 0.5_real64 * (wp(c) + wm(c))
            if (use_nn) then
                x(1) = p
                do j = 1, hid
                    x(1 + j) = hs(j, c); x(1 + hid + j) = cs(j, c)
                end do
                call bubbles_infer(x, y)
                src(c) = y(1)
                do j = 1, hid
                    hs(j, c) = y(1 + j); cs(j, c) = y(1 + hid + j)
                end do
            else
                src(c) = population_step(R(:, c), V(:, c), p)
            end if
        end do
        !$omp target teams distribute parallel do collapse(2) private(c, l, r_)
        do b = 1, nb_
            do i = 1, nx
                c = (b - 1) * nx + i
                l = (b - 1) * nx + max(i - 1, 1); r_ = (b - 1) * nx + min(i + 1, nx)
                wp_new(c) = wp(l) - dt * beta(c) * src(c)
                wm_new(c) = wm(r_) - dt * beta(c) * src(c)
            end do
        end do
    end subroutine

    subroutine run(wp, wm, wp2, wm2, beta, R, V, hs, cs, src, use_nn)
        real(real64), intent(inout) :: wp(ncell), wm(ncell), wp2(ncell), wm2(ncell), src(ncell)
        real(real64), intent(in) :: beta(ncell)
        real(real64), intent(inout) :: R(nbin, ncell), V(nbin, ncell), hs(hid, ncell), cs(hid, ncell)
        logical, intent(in) :: use_nn
        integer :: s
        do s = 1, nsteps / 2
            call step(wp, wm, wp2, wm2, beta, R, V, hs, cs, src, use_nn)
            call step(wp2, wm2, wp, wm, beta, R, V, hs, cs, src, use_nn)
        end do
    end subroutine

    pure real(real64) function hash01(a, b)
        integer, intent(in) :: a, b
        hash01 = real(mod((int(a, 8) * 40503_8 + int(b, 8)) * 2654435761_8, 4294967296_8), real64) &
                 / 4294967296.0_real64
    end function

    ! A right-going pulse per line, hashed amplitude and width.
    subroutine set_pulse(wp, wm)
        real(real64), intent(out) :: wp(ncell), wm(ncell)
        real(real64) :: amp, sig, x
        integer :: b, i
        do b = 1, nb_
            amp = 0.1_real64 + 0.25_real64 * hash01(b, 1); sig = 0.5_real64 + 1.0_real64 * hash01(b, 2)
            do i = 1, nx
                x = (i - 0.5_real64) * dx
                wp((b - 1) * nx + i) = 2.0_real64 * amp * exp(-0.5_real64 * ((x - 4.0_real64) / sig)**2)
                wm((b - 1) * nx + i) = 0.0_real64
            end do
        end do
    end subroutine
end program

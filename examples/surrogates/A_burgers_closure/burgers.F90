! Coarse-grid viscous Burgers with a learned subgrid closure, Fortran.
!
! The line-for-line twin of burgers.c: a per-cell closure called from the
! solver's own offload loop, weights embedded in the generated module (no
! init, nothing to load), the solver's arrays mapped once before the time
! loop, and nothing inside the loop allocating or transferring.
!
! One Fortran-specific point: the step loop reaches its arrays as
! explicit-shape dummies (u(n, NB)), never as allocatables. An allocatable
! referenced inside a target region carries a descriptor, and amdflang's
! OpenMP re-maps that descriptor on every region entry -- a small copy per
! array per time step, in a loop whose data is fully resident. Explicit
! shape has no descriptor, so the loop moves nothing.
program burgers
    use closure_model, only: closure_infer
    use iso_fortran_env, only: real64
    use omp_lib, only: omp_get_wtime
    implicit none
#ifndef NB
#define NB 64
#endif
    integer, parameter :: nb_ = NB               ! ensemble size; -DNB=4 for a quick host run
    integer, parameter :: nf = 2048, factor = 16, nc = nf / factor
    real(real64), parameter :: nu = 0.02_real64, dt_c = 0.01_real64
    integer, parameter :: n_sub = 64, nsteps = 200
    real(real64), parameter :: pi = 3.14159265358979323846_real64, L = 2.0_real64 * pi

    real(real64), allocatable :: uf(:,:), tf(:,:), ref(:,:), uc(:,:), tc(:,:), un(:,:), tn(:,:)
    real(real64) :: dxf, dxc, t0, t_ref, t_coarse, t_nn, e_coarse, e_nn, amp(3), ph(3), x
    integer :: b, i, k

    allocate(uf(nf, nb_), tf(nf, nb_), ref(nc, nb_), uc(nc, nb_), tc(nc, nb_), un(nc, nb_), tn(nc, nb_))
    dxf = L / nf; dxc = L / nc
    ! The same deterministic ensemble of initial conditions as burgers.c.
    do b = 1, nb_
        do k = 1, 3
            amp(k) = (2.0_real64 * hash01(b, 2 * k - 1) - 1.0_real64) / k
            ph(k) = 2.0_real64 * pi * hash01(b, 2 * k)
        end do
        do i = 1, nf
            x = (i - 1) * dxf
            uf(i, b) = sum([(amp(k) * sin(k * x + ph(k)), k = 1, 3)])
        end do
    end do
    call box_filter(uf, uc)
    un = uc

    ! 1. Fine reference: n_sub sub-steps per coarse step, then box-filter.
    t0 = omp_get_wtime()
    call run(uf, tf, nf, dxf, dt_c / n_sub, nsteps * n_sub, .false.)
    t_ref = omp_get_wtime() - t0
    call box_filter(uf, ref)
    ! 2. Coarse alone.  3. Coarse + closure.
    t0 = omp_get_wtime()
    call run(uc, tc, nc, dxc, dt_c, nsteps, .false.)
    t_coarse = omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call run(un, tn, nc, dxc, dt_c, nsteps, .true.)
    t_nn = omp_get_wtime() - t0

    e_coarse = mean_rel_l2(uc, ref); e_nn = mean_rel_l2(un, ref)
    print '(A,I0,A,I0,A,I0,A)', 'ensemble of ', nb_, ' realizations, ', nc, ' coarse cells, ', nsteps, &
        ' steps; mean relative L2 error vs the filtered fine reference:'
    print '(A,ES11.4,A,F8.1,A)', '  coarse scheme alone     ', e_coarse, '   (', 1e3 * t_coarse, ' ms)'
    print '(A,ES11.4,A,F8.1,A,F8.2,A)', '  coarse + NN closure     ', e_nn, '   (', 1e3 * t_nn, ' ms, ', &
        1e9 * (t_nn - t_coarse) / (real(nb_, real64) * nc * nsteps), ' ns per cell-step for the closure)'
    print '(A,I0,A,F8.1,A)', '  fine reference, ', nf, ' cells               (', 1e3 * t_ref, ' ms)'
    if (.not. (e_nn < e_coarse)) then
        print '(A)', 'FAIL: the closure did not improve on the coarse scheme'
        stop 1
    end if
    print '(A,F4.1,A)', 'OK: closure reduces the error by ', e_coarse / e_nn, 'x'

contains

    pure function godunov(ul, ur) result(f)
        !$omp declare target
        real(real64), intent(in) :: ul, ur
        real(real64) :: f, a, b
        a = 0.5_real64 * ul * ul; b = 0.5_real64 * ur * ur
        if (ul <= ur) then
            if (ul <= 0.0_real64 .and. ur >= 0.0_real64) then
                f = 0.0_real64
            else
                f = min(a, b)
            end if
        else
            f = max(a, b)
        end if
    end function

    ! One forward-Euler step of the scheme on every realization, into unew.
    ! With the closure on, NN(stencil) is added to each cell's right-hand
    ! side: this is the call a solver makes inside its step loop.
    subroutine step(u, unew, n, dx, dt, use_nn)
        integer, intent(in) :: n
        real(real64), intent(in) :: u(n, nb_), dx, dt
        real(real64), intent(out) :: unew(n, nb_)
        logical, intent(in) :: use_nn
        real(real64) :: rhs, stencil(7), corr(1)
        integer :: b, i, im3, im2, im1, ip1, ip2, ip3
        !$omp target teams distribute parallel do collapse(2) private(rhs, stencil, corr, im3, im2, im1, ip1, ip2, ip3)
        do b = 1, nb_
            do i = 1, n
                im3 = modulo(i - 4, n) + 1; im2 = modulo(i - 3, n) + 1; im1 = modulo(i - 2, n) + 1
                ip1 = modulo(i, n) + 1; ip2 = modulo(i + 1, n) + 1; ip3 = modulo(i + 2, n) + 1
                rhs = -(godunov(u(i, b), u(ip1, b)) - godunov(u(im1, b), u(i, b))) / dx &
                    + nu * (u(ip1, b) - 2.0_real64 * u(i, b) + u(im1, b)) / (dx * dx)
                if (use_nn) then
                    stencil = [u(im3, b), u(im2, b), u(im1, b), u(i, b), u(ip1, b), u(ip2, b), u(ip3, b)]
                    call closure_infer(stencil, corr)      ! the surrogate, per cell, on the device
                    rhs = rhs + corr(1)
                end if
                unew(i, b) = u(i, b) + dt * rhs
            end do
        end do
    end subroutine

    ! Map once; the step loop moves nothing. The two mapped buffers
    ! alternate roles on the device; the result ends in u.
    subroutine run(u, tmp, n, dx, dt, nsteps_, use_nn)
        integer, intent(in) :: n, nsteps_
        real(real64), intent(inout) :: u(n, nb_), tmp(n, nb_)
        real(real64), intent(in) :: dx, dt
        logical, intent(in) :: use_nn
        integer :: s
        !$omp target enter data map(to: u) map(alloc: tmp)
        do s = 1, nsteps_ / 2
            call step(u, tmp, n, dx, dt, use_nn)
            call step(tmp, u, n, dx, dt, use_nn)
        end do
        if (mod(nsteps_, 2) == 1) then
            call step(u, tmp, n, dx, dt, use_nn)
            !$omp target update from(tmp)
            u = tmp
            !$omp target update to(u)
        end if
        !$omp target exit data map(from: u) map(delete: tmp)
    end subroutine

    subroutine box_filter(fine, coarse)
        real(real64), intent(in) :: fine(nf, nb_)
        real(real64), intent(out) :: coarse(nc, nb_)
        integer :: b, i
        do b = 1, nb_
            do i = 1, nc
                coarse(i, b) = sum(fine((i - 1) * factor + 1:i * factor, b)) / factor
            end do
        end do
    end subroutine

    ! Mean over the ensemble of the relative L2 error of each realization.
    function mean_rel_l2(a, r) result(m)
        real(real64), intent(in) :: a(nc, nb_), r(nc, nb_)
        real(real64) :: m
        integer :: b
        m = 0.0_real64
        do b = 1, nb_
            m = m + sqrt(sum((a(:, b) - r(:, b))**2) / sum(r(:, b)**2))
        end do
        m = m / nb_
    end function

    ! The integer hash burgers.c uses; every intermediate fits a signed int64.
    function hash01(b, k) result(v)
        integer, intent(in) :: b, k
        real(real64) :: v
        v = real(mod((int(b, 8) * 40503_8 + int(k, 8)) * 2654435761_8, 4294967296_8), real64) &
            / 4294967296.0_real64
    end function
end program

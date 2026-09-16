! Calling a generated roseNNa model from Fortran.
!
! Built by run_basic.sh, which generates gemm_small first. gemm_small embeds its
! weights, so there is no gemm_small_init to call; a larger model would call
!     call gemm_small_init("gemm_small.rwt", status)
! once before the first infer.
program capiTester
    use gemm_small_model
    use iso_fortran_env, only: real32
    implicit none

    ! One point: n_in = 2 values in, n_out = 3 values out.
    real(real32) :: x(2), y(3)

    x = 1.0_real32
    call gemm_small_infer(x, y)

    print '(3(f0.6,1x))', y
end program

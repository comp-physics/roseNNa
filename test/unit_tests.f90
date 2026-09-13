program unit_tests
    use iso_c_binding
    use activation_functions
    use derived_types
    use model_layers
    implicit none

    integer :: failures = 0

    call test_relu_basic()
    call test_sigmoid_midpoint()

    if (failures > 0) then
        write(*,'(a,i0,a)') 'UNIT TESTS: ', failures, ' failure(s)'
        stop 1
    end if
    write(*,'(a)') 'UNIT TESTS: all passed'

contains

    subroutine check(cond, name)
        logical, intent(in) :: cond
        character(*), intent(in) :: name
        if (cond) then
            write(*,'(a,a)') '  ok   ', name
        else
            write(*,'(a,a)') '  FAIL ', name
            failures = failures + 1
        end if
    end subroutine

    subroutine test_relu_basic()
        real(c_double) :: x(3), y(3)
        x = [-1.0d0, 0.0d0, 2.0d0]
        y = relu(x)
        call check(all(abs(y - [0.0d0, 0.0d0, 2.0d0]) < 1.0d-12), 'relu clamps negatives')
    end subroutine

    subroutine test_sigmoid_midpoint()
        real(c_double) :: y(1)
        y = sigmoid([0.0d0])
        call check(abs(y(1) - 0.5d0) < 1.0d-12, 'sigmoid(0) == 0.5')
    end subroutine

end program unit_tests

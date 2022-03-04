module derived_types

    use activation_functions

    implicit none

    abstract interface
        function func (z) result(output)
            real, intent(in) :: z(:)
            real :: output(size(z))
        end function func
    end interface
    
    TYPE linLayer
        REAL, ALLOCATABLE, DIMENSION(:,:) :: weights
        REAL, ALLOCATABLE, DIMENSION(:) :: biases
        procedure(func), POINTER, NOPASS :: fn_ptr => null ()
    ENDTYPE linLayer

    

    
end module derived_types
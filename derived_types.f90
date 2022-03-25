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

    TYPE lstmLayer
        REAL, ALLOCATABLE, DIMENSION(:,:) :: whh
        REAL, ALLOCATABLE, DIMENSION(:,:) :: wih
        REAL, ALLOCATABLE, DIMENSION(:) :: bhh
        REAL, ALLOCATABLE, DIMENSION(:) :: bih
    ENDTYPE lstmLayer

    

    
end module derived_types
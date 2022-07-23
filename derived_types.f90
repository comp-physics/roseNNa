module derived_types

    use activation_functions

    implicit none

    abstract interface
        function func (z) result(output)
            real, intent(in) :: z(:,:)
            real :: output(size(z,1), size(z,2))
        end function func
    end interface
    
    TYPE linLayer
        REAL, ALLOCATABLE, DIMENSION(:,:) :: weights
        REAL, ALLOCATABLE, DIMENSION(:) :: biases
        procedure(func), POINTER, NOPASS :: fn_ptr => null ()
    ENDTYPE linLayer

    TYPE lstmLayer
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: whh
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: wih
        REAL, ALLOCATABLE, DIMENSION(:) :: bhh
        REAL, ALLOCATABLE, DIMENSION(:) :: bih
    ENDTYPE lstmLayer

    TYPE convLayer
        REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: weights
        REAL, ALLOCATABLE, DIMENSION(:) :: biases
        !==stride
    ENDTYPE convLayer

    TYPE maxpoolLayer
        INTEGER :: kernel_size
    ENDTYPE maxpoolLayer

    TYPE avgpoolLayer
        INTEGER :: kernel_size
    ENDTYPE avgpoolLayer

    TYPE addLayer
        REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: adder
    ENDTYPE addLayer

    TYPE reshapeLayer
        REAL, ALLOCATABLE, DIMENSION(:,:) :: reshape2d
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: reshape3d
        REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: reshape4d
    ENDTYPE

    
end module derived_types
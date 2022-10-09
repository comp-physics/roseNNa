module derived_types
    use iso_c_binding
    use activation_functions

    implicit none

    abstract interface
        function func (z) result(output)
            REAL, intent(in) :: z(:,:)
            real :: output(size(z,1), size(z,2))
        end function func
    end interface
    
    TYPE linLayer
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: weights
        REAL (c_double), ALLOCATABLE, DIMENSION(:) :: biases
    ENDTYPE linLayer

    TYPE lstmLayer
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: whh
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: wih
        REAL (c_double), ALLOCATABLE, DIMENSION(:) :: bhh
        REAL (c_double), ALLOCATABLE, DIMENSION(:) :: bih
    ENDTYPE lstmLayer

    TYPE convLayer
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: weights
        REAL (c_double), ALLOCATABLE, DIMENSION(:) :: biases
        !==stride
    ENDTYPE convLayer

    TYPE maxpoolLayer
        INTEGER :: kernel_size
    ENDTYPE maxpoolLayer

    TYPE avgpoolLayer
        INTEGER :: kernel_size
    ENDTYPE avgpoolLayer

    TYPE addLayer
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: adder
    ENDTYPE addLayer

    TYPE reshapeLayer
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: reshape2d
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: reshape3d
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: reshape4d
    ENDTYPE

    
end module derived_types
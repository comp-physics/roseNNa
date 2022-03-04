PROGRAM linear
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    ! ===============================================================

    
    IMPLICIT NONE


    INTERFACE
        FUNCTION linear_layer(inp, weights, bias) result(output)
            real :: output(size(weights, dim=1))
            real, intent(in) :: weights(:,:)
            real, intent(in) :: inp(:)
            real, intent(in) :: bias(:)
        END FUNCTION linear_layer
    END INTERFACE
    
    REAL :: inp(2)
    REAL, ALLOCATABLE :: out(:)
    INTEGER :: i
    REAL :: T1, T2
    inp = reshape ((/ 1.0 , 1.0  /), shape(inp))

    
    
    ALLOCATE(out(10))
    CALL CPU_TIME(T1)
    CALL init()
    out = inp
    DO i = 1, SIZE(layers)
        out = linear_layer(out, layers(i)%weights, layers(i)%biases)
        out = layers(i)%fn_ptr(out)
    END DO
    call CPU_TIME(T2)

    print *, out
    print *, T2-T1
    
    

END PROGRAM linear

!=======================================================================================

FUNCTION linear_layer(inp, weights, bias) result(output)
    IMPLICIT NONE

    real, intent(in) :: weights(:,:)
    real, intent(in) :: inp(:)
    real, intent(in) :: bias(:)
    real :: output(size(weights, dim=1))

    output = matmul(weights,inp) + bias
END FUNCTION linear_layer

subroutine lstm_cell(input, hid1, cell1, Whh, Wih, Bih, Bhh, hiddenOut, cellOut)
    implicit none
    real, intent(in), DIMENSION(:) :: input !== (n,1)
    real, intent(in), DIMENSION(:) :: hid1 !==(m,1)
    real, intent(in), DIMENSION(:) :: cell1 !==(m,1)
    real, intent(in), DIMENSION(:,:) :: Whh !==(4m,m)
    real, intent(in), DIMENSION(:,:) :: Wih !==(4m,n)
    real, intent(in), DIMENSION(:) :: Bhh !==(4m,1)
    real, intent(in), DIMENSION(:) :: Bih !==(4m,1)
    real, intent(out), DIMENSION(size(hid1)) :: hiddenOut
    real, intent(out), DIMENSION(size(cell1)) :: cellOut

    real, DIMENSION(size(Whh, dim=1)) :: gates_out
    integer :: chunk

    chunk = size(Whh, dim=1) / 4
    gates_out = matmul(input, Wih) + Bih + matmul(hid1, Whh) + Bhh

    gates_out(1:chunk) = 
    gates_out(chunk:chunk*2)



end
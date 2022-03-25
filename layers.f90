module model_layers

    USE activation_functions

    implicit none

contains
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
        real, ALLOCATABLE, DIMENSION(:), intent(out) :: hiddenOut
        real, ALLOCATABLE, DIMENSION(:), intent(out) :: cellOut

        real, DIMENSION(size(Whh, dim=1)) :: gates_out
        real, DIMENSION(4,size(hid1)) :: chunks !4, size(Whh, dim=1)/4
        ALLOCATE(hiddenOut(size(hid1)))
        ALLOCATE(cellOut(size(cell1)))

        gates_out = linear_layer(input, Wih, Bih) + linear_layer(hid1, Whh, Bhh) !==(4m,1)
        chunks = RESHAPE(gates_out, (/ 4, size(gates_out)/4 /))        

        chunks(1, :) = sigmoid(chunks(1, :))
        chunks(2, :) = sigmoid(chunks(2, :))
        chunks(3, :) = tanhh(chunks(3, :))
        chunks(4, :) = sigmoid(chunks(4, :))
        
        hiddenOut = chunks(4, :) * tanhh((chunks(2, :) * cell1) + (chunks(1, :) * chunks(3, :)))
        cellOut = (chunks(2, :) * cell1) + (chunks(1, :) * chunks(3, :))
    end subroutine
end module model_layers
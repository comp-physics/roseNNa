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
        real, intent(out), DIMENSION(size(hid1)) :: hiddenOut
        real, intent(out), DIMENSION(size(cell1)) :: cellOut

        real, DIMENSION(size(Whh, dim=1)) :: gates_out
        real, DIMENSION(4, size(Whh, dim=1)/4) :: chunks
        ! real, DIMENSION(size(Whh, dim=1)/4) :: ingate
        ! real, DIMENSION(size(Whh, dim=1)/4) :: forget
        ! real, DIMENSION(size(Whh, dim=1)/4) :: cell
        ! real, DIMENSION(size(Whh, dim=1)/4) :: out


        integer :: chunk
        integer :: i

        chunk = size(Whh, dim=1) / 4
        gates_out = linear_layer(input, Wih, Bih) + linear_layer(hid1, Whh, Bhh) !==(4m,1)
        CALL get_chunks(gates_out, chunks)
        chunks(1, :) = sigmoid(chunks(1, :))
        chunks(2, :) = sigmoid(chunks(2, :))
        chunks(3, :) = tanh(chunks(3, :))
        chunks(4, :) = sigmoid(chunks(4, :))

        cellOut = (chunks(2, :) * cell1) + (chunks(1, :) * chunks(3, :))
        hiddenOut = chunks(4, :) * tanh(cellOut)



    
    
    end subroutine

    subroutine get_chunks(largeChunk, chunks)
        implicit none

        real, intent(in) :: largeChunk(:)
        real, intent(out), DIMENSION(4, size(largeChunk)/4) :: chunks
        integer :: chunk
        integer :: i
        chunk = size(largeChunk)/4
        DO i = 0, size(largeChunk)-chunk, chunk
            chunks(i/chunk, :) = largeChunk(i:i+chunk)
        END DO



    end subroutine





end module model_layers
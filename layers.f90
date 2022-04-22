module model_layers

    USE activation_functions
    USE derived_types

    implicit none

contains
    subroutine linear_layer(inp, lin) 
        IMPLICIT NONE
        real, ALLOCATABLE, intent(inout) :: inp(:,:)
        TYPE(linLayer), INTENT(IN) :: lin
        ! real, intent(inout) :: output(:,:)
        real, DIMENSION(size(inp,1)) :: inpReshape
        INTEGER :: outShape
        outShape = size(lin%weights, 1)
        inpReshape = RESHAPE(inp, SHAPE(inpReshape))
        inp = RESHAPE(lin%fn_ptr(matmul(lin%weights,inpReshape) + lin%biases), (/outShape, 1/))
    end subroutine 

    subroutine lstm_cell(input, hid1, cell1, Whh, Wih, Bih, Bhh, hiddenOut, cellOut)
        implicit none
        real, intent(in), DIMENSION(:,:) :: input !== (n,batch_size)
        real, intent(in), DIMENSION(:,:) :: hid1 !==(m,batch_size)
        real, intent(in), DIMENSION(:,:) :: cell1 !==(m,batch_size)
        real, intent(in), DIMENSION(:,:) :: Whh !==(4m,m)
        real, intent(in), DIMENSION(:,:) :: Wih !==(4m,n)
        real, intent(in), DIMENSION(:) :: Bhh !==(4m,1)
        real, intent(in), DIMENSION(:) :: Bih !==(4m,1)
        real, ALLOCATABLE, DIMENSION(:,:), intent(out) :: hiddenOut !==(m,batch_size)
        real, ALLOCATABLE, DIMENSION(:,:), intent(out) :: cellOut


        real, DIMENSION(size(Bhh, 1), size(input,2)) :: Bhh_broadcast !== (4m, batch_size)
        real, DIMENSION(size(Bhh, 1), size(input,2)) :: Bih_broadcast
        real, DIMENSION(size(Whh, dim=1), size(input,2)) :: gates_out
        real, DIMENSION(size(hid1,1),4,size(input,2)) :: chunks !4, size(Whh, dim=1)/4
        ALLOCATE(hiddenOut(size(hid1),size(input,2)))
        ALLOCATE(cellOut(size(cell1),size(input,2)))
        Bhh_broadcast = SPREAD(Bhh, 2, size(input,2))
        Bih_broadcast = SPREAD(Bih, 2, size(input,2))

        gates_out = MATMUL(Wih, input) + Bih_broadcast + MATMUL(Whh, hid1) + Bhh_broadcast !==(4m,batch_size) BROADCAST BIAS
        chunks = RESHAPE(gates_out, (/size(gates_out,1)/4, 4, size(input,2)/))

        chunks(:, 1, :) = sigmoid2d(chunks(:, 1, :))
        !RESHAPE(sigmoid(RESHAPE(chunks(:, 1, :), (/size(gates_out,1)/4*size(input,2)/))), &
                                                                    !SHAPE(chunks(1,:,:)))
        chunks(:, 2, :) = sigmoid2d(chunks(:, 2, :))
        !RESHAPE(sigmoid(RESHAPE(chunks(:, 1, :), (/size(gates_out,1)/4*size(input,2)/))), &
                                                                    !SHAPE(chunks(2,:,:))) !== (m, batch_size)
        chunks(:, 3, :) = tanhh2d(chunks(:, 3, :))
        !RESHAPE(tanhh(RESHAPE(chunks(:, 1, :), (/size(gates_out,1)/4*size(input,2)/))), &
                                                                    !SHAPE(chunks(3,:,:)))
        chunks(:, 4, :) = sigmoid2d(chunks(:, 4, :))
        !RESHAPE(sigmoid(RESHAPE(chunks(:, 1, :), (/size(gates_out,1)/4*size(input,2)/))), &
                                                                    !SHAPE(chunks(4,:,:)))
        cellOut = (chunks(:, 2, :) * cell1) + (chunks(:, 1, :) * chunks(:, 3, :))
        hiddenOut = chunks(:, 4, :) * tanhh2d(cellOut)
    end subroutine

    subroutine lstm(input, hid1, cell1, Whh, Wih, Bih, Bhh, hiddenOut, cellOut)
        implicit none
        real, intent(in), DIMENSION(:,:,:) :: input !== (timesteps,n,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: hid1 !==(m,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: cell1 !==(m,batch_size)
        real, intent(in), ALLOCATABLE, DIMENSION(:,:) :: Whh !==(4m,m)
        real, intent(in), ALLOCATABLE, DIMENSION(:,:) :: Wih !==(4m,n)
        real, intent(in), ALLOCATABLE, DIMENSION(:) :: Bhh !==(4m,1)
        real, intent(in), ALLOCATABLE, DIMENSION(:) :: Bih !==(4m,1)
        ! INTEGER, INTENT(IN) :: nlayers, NEED TO ADD NLAYERS FUNCTIONALITY
        real, ALLOCATABLE, DIMENSION(:,:), intent(out) :: hiddenOut !==(m,batch_size)
        real, ALLOCATABLE, DIMENSION(:,:), intent(out) :: cellOut
        INTEGER :: timesteps 
        INTEGER :: i
        timesteps = size(input,1)

        DO i=1, timesteps
            CALL lstm_cell(input(i,:,:), hid1, cell1, Whh, Wih, Bih, Bhh, hiddenOut, cellOut)
            print *, hiddenOut
            hid1 = hiddenOut
            cell1 = cellOut
        END DO
    end subroutine

    subroutine conv(inp, convWeights, bias, in_channels, out_channels, kernel_size, stride, out)
        implicit none
        REAL, INTENT(IN), DIMENSION(:,:,:) :: inp !==(numImages,imageD1,imageD2)
        REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:,:,:,:) :: convWeights !==(numConvRows,numConvCols,ConvRowDim,ConvColDim)
        REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:) :: bias
        INTEGER, INTENT(IN) :: in_channels !==numImages
        INTEGER, INTENT(IN) :: out_channels !==numConvCols
        INTEGER, INTENT(IN) :: kernel_size !==(ConvRowDim,ConvColDim)
        INTEGER, INTENT(IN) :: stride
        REAL, ALLOCATABLE, DIMENSION(:,:,:), INTENT(OUT) :: out
        
        INTEGER :: outer
        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        REAL :: sumini = 0
        ALLOCATE(out(out_channels, size(inp,dim=2)-kernel_size + 1, size(inp,dim=2)-kernel_size+1))
        outRowDim = size(inp,dim=2)-kernel_size + 1
        outColDim = size(inp,dim=2)-kernel_size + 1

        DO outer = 0, out_channels-1 !==iterating through each output image
            DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
                DO inner = 0, in_channels-1 !==applying kernel to each input image
                    sumini = sumini + SUM(inp(inner+1,(overImage/outRowDim + 1):(overImage/outRowDim+kernel_size) &
                                      ,(MODULO(overImage,outColDim) + 1):(MODULO(overImage,outColDim)+kernel_size)) &
                          * convWeights(outer+1,inner+1,:,:)) !==depending on the way convWeights is laid out change position of outer/inner
                END DO
                out(outer+1,overImage/outRowDim + 1,MODULO(overImage,outColDim)+1) = sumini + bias(outer+1)
                sumini = 0;
            END DO
        END DO

    END subroutine


    subroutine max_pool(inp, kernel_size, stride, out)
        implicit none
        REAL, INTENT(IN), DIMENSION(:,:,:) :: inp !==(numImages,imageD1,imageD2)
        INTEGER, INTENT(IN) :: kernel_size !==(ConvRowDim,ConvColDim)
        INTEGER, INTENT(IN) :: stride
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: out
        
        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        ALLOCATE(out(size(inp,dim=1), size(inp,dim=2)/kernel_size, size(inp,dim=3)/kernel_size))
        outRowDim = size(inp,dim=2)/kernel_size
        outColDim = size(inp,dim=3)/kernel_size

        DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
            DO inner = 0, size(inp,dim=1)-1 !==applying kernel to each input image
                out(inner+1,(overImage/outRowDim)+1,MODULO(overImage,outColDim)+1) = &
                 MAXVAL(inp(inner+1,((overImage/outRowDim)*kernel_size + 1):((overImage/outRowDim)*kernel_size)+kernel_size &
                 ,(MODULO(overImage,outColDim)*kernel_size + 1): &
                (MODULO(overImage,outColDim)*kernel_size+kernel_size)))
            END DO
        END DO

    end subroutine


    
end module model_layers
module model_layers

    USE activation_functions
    USE derived_types

    implicit none

contains
    !=== for basic MM operations ======
    subroutine linear_layer(inp, lin) 
        IMPLICIT NONE
        real, ALLOCATABLE, intent(inout) :: inp(:,:) !===input is 2d usually (m,1)
        TYPE(linLayer), INTENT(IN) :: lin !===stores the weights and biases
        real, DIMENSION(size(inp,1)) :: inpReshape !===to reshape input to 1d
        INTEGER :: outShape
        outShape = size(lin%weights, 1)
        inpReshape = RESHAPE(inp, SHAPE(inpReshape))
        inp = RESHAPE(lin%fn_ptr(matmul(lin%weights,inpReshape) + lin%biases), (/outShape, 1/)) !====reshape to 1d, do operations, then reshape back to 2d
    end subroutine 

    subroutine lstm_cell(input, hid1, cell1, Whh, Wih, Bih, Bhh)
        implicit none
        real, intent(in), DIMENSION(:,:) :: input !== (n,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: hid1 !==(m,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: cell1 !==(m,batch_size)
        real, intent(in), DIMENSION(:,:) :: Whh !==(4m,m)
        real, intent(in), DIMENSION(:,:) :: Wih !==(4m,n)
        real, intent(in), DIMENSION(:) :: Bhh !==(4m,1)
        real, intent(in), DIMENSION(:) :: Bih !==(4m,1)
        real, ALLOCATABLE, DIMENSION(:,:) :: hiddenOut !==(m,batch_size)
        real, ALLOCATABLE, DIMENSION(:,:) :: cellOut


        real, DIMENSION(size(Bhh, 1), size(input,2)) :: Bhh_broadcast !== (4m, batch_size)
        real, DIMENSION(size(Bhh, 1), size(input,2)) :: Bih_broadcast
        real, DIMENSION(size(Whh, dim=1), size(input,2)) :: gates_out
        real, DIMENSION(size(hid1,1),4,size(input,2)) :: chunks !4, size(Whh, dim=1)/4
        ALLOCATE(hiddenOut(size(hid1),size(input,2)))
        ALLOCATE(cellOut(size(cell1),size(input,2)))

        !======= applying bhh and bih to each column of input =====
        Bhh_broadcast = SPREAD(Bhh, 2, size(input,2)) 
        Bih_broadcast = SPREAD(Bih, 2, size(input,2))

        gates_out = MATMUL(Wih, input) + Bih_broadcast + MATMUL(Whh, hid1) + Bhh_broadcast !==(4m,batch_size) BROADCAST BIAS
        chunks = RESHAPE(gates_out, (/size(gates_out,1)/4, 4, size(input,2)/))

        chunks(:, 1, :) = sigmoid2d(chunks(:, 1, :))
        chunks(:, 2, :) = sigmoid2d(chunks(:, 2, :))
        chunks(:, 3, :) = tanhh2d(chunks(:, 3, :))
        chunks(:, 4, :) = sigmoid2d(chunks(:, 4, :))
        
        cellOut = (chunks(:, 2, :) * cell1) + (chunks(:, 1, :) * chunks(:, 3, :))
        hiddenOut = chunks(:, 4, :) * tanhh2d(cellOut)
        hid1 = hiddenOut
        cell1 = cellOut
    end subroutine

    subroutine lstm(input, hid1, cell1, Whh, Wih, Bih, Bhh)
        implicit none
        real, intent(in), DIMENSION(:,:,:) :: input !== (timesteps,n,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: hid1 !==(m,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: cell1 !==(m,batch_size)
        real, intent(in), ALLOCATABLE, DIMENSION(:,:) :: Whh !==(4m,m)
        real, intent(in), ALLOCATABLE, DIMENSION(:,:) :: Wih !==(4m,n)
        real, intent(in), ALLOCATABLE, DIMENSION(:) :: Bhh !==(4m,1)
        real, intent(in), ALLOCATABLE, DIMENSION(:) :: Bih !==(4m,1)
        ! INTEGER, INTENT(IN) :: nlayers, NEED TO ADD NLAYERS FUNCTIONALITY
        INTEGER :: timesteps 
        INTEGER :: i
        timesteps = size(input,1)

        DO i=1, timesteps
            CALL lstm_cell(input(i,:,:), hid1, cell1, Whh, Wih, Bih, Bhh)
            print *, hid1
        END DO
        hid1 = reshape(hid1, (/SIZE(hid1,dim=2), SIZE(hid1,dim=1)/), order = [2,1]) !==hid1 is usually output, and is now shape (batch_size,m), this is mainly for onnx functionality
    end subroutine

    subroutine conv(inp, convWeights, bias) !=in_channels, out_channels, kernel_size, stride
        implicit none
        REAL, INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:) :: inp !==(numImages,imageD1,imageD2)
        REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:,:,:,:) :: convWeights !==(numConvRows,numConvCols,ConvRowDim,ConvColDim)
        REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:) :: bias
        INTEGER :: in_channels !==numImages SHOULD BE INTENT(IN)
        INTEGER :: out_channels !==numConvCols SHOULD BE INTENT(IN)
        INTEGER :: kernel_size !==(ConvRowDim,ConvColDim) SHOULD BE INTENT(IN)
        INTEGER :: stride !===IMPLEMENT LATER
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: out
        
        INTEGER :: outer
        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        REAL :: sumini = 0
        in_channels = SIZE(inp, dim=1)
        out_channels = SIZE(convWeights, dim=2)
        kernel_size = SIZE(convWeights, dim=3)
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
        inp = out
        DEALLOCATE(out)

    END subroutine


    subroutine max_pool(inp, maxpool) !==stride
        implicit none
        REAL, INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:) :: inp !==(numImages,imageD1,imageD2)
        TYPE(maxpoolLayer), INTENT(IN) :: maxpool !==(ConvRowDim,ConvColDim)
        !==INTEGER, INTENT(IN) :: stride IMPLEMENT THIS
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: out
        INTEGER :: kernel_size
        
        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        kernel_size = maxpool%kernel_size
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
        inp = out
        DEALLOCATE(out)

    end subroutine


    
end module model_layers
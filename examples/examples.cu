// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.cuh"

using namespace std;
using namespace troy;

int main()
{
    while (true)
    {
        cout << "+---------------------------------------------------------+" << endl;
        cout << "| The following examples should be executed while reading |" << endl;
        cout << "| comments in associated files in native/examples/.       |" << endl;
        cout << "+---------------------------------------------------------+" << endl;
        cout << "| Examples                   | Source Files               |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;
        cout << "| 1. BFV Basics              | 1_bfv_basics.cpp           |" << endl;
        cout << "| 2. Encoders                | 2_encoders.cpp             |" << endl;
        cout << "| 3. Levels                  | 3_levels.cpp               |" << endl;
        cout << "| 4. BGV Basics              | 4_bgv_basics.cpp           |" << endl;
        cout << "| 5. CKKS Basics             | 5_ckks_basics.cpp          |" << endl;
        cout << "| 6. Rotation                | 6_rotation.cpp             |" << endl;
        cout << "| 7. Serialization           | 7_serialization.cpp        |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;

        int selection = 0;
        bool valid = true;
        do
        {
            cout << endl << "> Run example by typing number, or 0 to exit: ";
            if (!(cin >> selection))
            {
                valid = false;
            }
            else
            {
                valid = true;
            }
            if (!valid)
            {
                cout << "  [Beep~~] valid option." << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        } while (!valid);

        switch (selection)
        {
        case 1:
            example_bfv_basics();
            break;

        case 2:
            example_encoders();
            break;

        case 3:
            example_levels();
            break;

        case 4:
            example_bgv_basics();
            break;

        case 5:
            example_ckks_basics();
            break;

        case 6:
            example_rotation();
            break;

        case 7:
            example_serialization();
            break;

        case 0:
            troy::utils::MemoryPool::Destroy();
            return 0;
        }
    }

    return 0;
}
